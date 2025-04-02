import torch
from NER_cadec import EntityMatrixPredictor
from NER_cadec import get_train_loader
import torch.nn as nn

def training_loop_rl(dataset_dir, model=EntityMatrixPredictor(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epochs=3, pos_weight=15, verbose=False, subset_size=None, rl_weight=1):
    model.to(device)
    loss_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-05)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loader = get_train_loader(dataset_dir, subset_size=subset_size)
        for batch in train_loader:
            tokens = batch[0]
            target_matrix = batch[1].to(device)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            word_ids = [tokens.word_ids(batch_index=i) for i in range(len(input_ids))]
            optimizer.zero_grad()
            predicted_matrix = model(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids)
            _, max_words, _ = predicted_matrix.shape
            target_matrix = target_matrix[:, :max_words, :max_words]
            rewards = []
            log_probs = []
            probs = torch.sigmoid(predicted_matrix)
            for i in range(probs.size(0)): 
                sampled_matrix = torch.bernoulli(probs[i]).detach().cpu()
                reward = loop_penalty(sampled_matrix)
                if verbose:
                    print(f"Reward: {reward}")
                rewards.append(reward)
                log_p = (sampled_matrix * torch.log(probs[i] + 1e-6) +
                        (1 - sampled_matrix) * torch.log(1 - probs[i] + 1e-6)).mean()
                log_probs.append(log_p)
            rewards = torch.tensor(rewards, device=probs.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            rl_penalty = -torch.stack(log_probs) @ rewards
            scaling_factor = (loss_bce(predicted_matrix, target_matrix).detach() / (rl_penalty + 1e-6))
            rl_penalty_scaled = rl_penalty * scaling_factor
            loss = loss_bce(predicted_matrix, target_matrix) + rl_weight * rl_penalty_scaled
            loss.backward()
            optimizer.step()
            if verbose:
                print(f'Batch loss {loss.item()}')
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        if verbose:
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
    return model

def reachability(adj_matrix):
    """
    Compute which nodes are in a cycle by checking diagonal entries of matrix powers.
    """
    n = adj_matrix.shape[0]
    reach = torch.zeros(n, dtype=torch.bool, device=adj_matrix.device)
    power = adj_matrix.clone()

    for _ in range(1, n + 1):
        reach |= torch.diag(power).bool()
        power = torch.matmul(power, adj_matrix)

    return reach

def nodes_with_arcs(adj_matrix):
    return (adj_matrix.sum(dim=0) > 0) | (adj_matrix.sum(dim=1) > 0)

def loop_penalty(pred_matrix):
    """
    Compute loop reward as the fraction of nodes involved in valid loops.

    Args:
        pred_matrix (torch.Tensor): NxN binary matrix (0/1) representing entity arcs.

    Returns:
        torch.Tensor: Scalar reward between 0 and 1.
    """
    pred_binary = (pred_matrix > 0.5).int()
    upper_triangle = torch.triu(pred_binary)
    lower_positions = (torch.tril(pred_binary, diagonal=-1) == 1).nonzero(as_tuple=False)

    n = pred_binary.shape[0]
    global_loop_flags = torch.zeros(n, dtype=torch.bool, device=pred_matrix.device)

    if lower_positions.numel() == 0:
        return 1 - pred_matrix.sum() / n

    for i, j in lower_positions:
        submatrix = upper_triangle[j:i+1, j:i+1].clone()
        submatrix[i - j, 0] = 1  # Insert loop-closing arc
        local_loop_flags = reachability(submatrix)  # shape: (i-j+1,)
        global_indices = torch.arange(j, i + 1, device=pred_matrix.device)
        global_loop_flags[global_indices] |= local_loop_flags

    nodes_with_edges = nodes_with_arcs(pred_binary)
    valid_nodes = ~(global_loop_flags ^ (~nodes_with_edges))  # XNOR
    num_valid = valid_nodes.sum().float()

    return (num_valid / n - 0.5)

def extract_spans_from_matrix(matrix):
    """
    Extracts and merges entity spans from upper triangular part of a matrix.

    Args:
        matrix (torch.Tensor): Binary entity matrix (max_words x max_words).

    Returns:
        List of (start, end) spans.
    """
    max_words = matrix.shape[0]
    spans = []

    for i in range(max_words):
        if matrix[i, i] == 1:
            spans.append([i, i])

    start = -1
    for i in range(max_words - 1):
        if matrix[i, i + 1] == 1:
            if start == -1:
                start = i
        elif start != -1:
            spans.append([start, i])
            start = -1
    if start != -1:
        spans.append([start, max_words - 1])

    return spans
