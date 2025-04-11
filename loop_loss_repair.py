import torch
import torch.nn as nn

from NER_cadec import EntityMatrixPredictor
from NER_cadec import get_train_loader

def training_loop_repair(
    dataset_dir,
    model=EntityMatrixPredictor(),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    epochs=3,
    pos_weight=15,
    verbose=False,
    subset_size=None,
    repair_loss_weight=1.0
):
    model.to(device)
    loss_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

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

            predicted_matrix = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids
            )

            _, max_words, _ = predicted_matrix.shape
            target_matrix = target_matrix[:, :max_words, :max_words]

            # Supervised loss
            supervised_loss = loss_bce(predicted_matrix, target_matrix)

            # Repair-based structural loss
            repair_loss = torch.stack([
                compute_repair_loss(predicted_matrix[i])
                for i in range(predicted_matrix.size(0))
            ]).mean()

            # Combined loss
            loss = supervised_loss + repair_loss_weight * repair_loss
            loss.backward()
            optimizer.step()

            if verbose:
                print(f'Batch loss {loss.item()} | BCE: {supervised_loss.item()} | Repair: {repair_loss.item()}')

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        if verbose:
            print(f'Epoch {epoch + 1}, Avg Loss: {epoch_loss:.4f}')

    return model


def find_opening_paths(matrix, threshold=0.5, max_distance=10):
    """
    Recursively find all maximal opening paths (as sequences of arcs) in the upper triangle.

    Args:
        matrix (torch.Tensor): A square matrix of shape (N, N) containing scores.
        threshold (float): Minimum value to consider an arc active.

    Returns:
        List[List[Tuple[int, int]]]: List of paths, where each path is a list of (i, j) arcs.
    """
    n = matrix.size(0)
    all_paths = []

    def recurse(path, distance=0):
        i,j = path[-1]
        branches = []
        for k in range(i + 1, min(i + distance,n)):
            if matrix[j, k] > threshold: 
                for m in range(k, min(i + distance,n)):
                    if matrix[j, m] > threshold: 
                        branches.append((j, m))
                        matrix[j, m] = 0
                break
        if branches == []: 
            all_paths.append(path)
            return
        for j,m in branches:
            recurse(path + [(j,m)], distance=distance-m)

    for i in range(n):
        for j in range(i + 1, min(i + max_distance, n)):
            if matrix[i, j] > threshold:
                recurse([(i, j)], distance=max_distance)

    return all_paths


def ordered_partitions(lst):
    """
    Generate all ordered partitions of a list.

    Example: [1,2] → [[[1], [2]], [[1,2]]]
    """
    if not lst:
        yield []
        return
    for i in range(1, len(lst) + 1):
        for rest in ordered_partitions(lst[i:]):
            yield [lst[:i]] + rest


def opening_paths_partitions(path):
    """
    Given a forward path (e.g., [(0,1),(1,2),(2,3)]), return all valid backward repairs
    as partitions closing arcs like (end, start).

    Args:
        path: List of forward arcs.

    Returns:
        List[List[Tuple[int, int]]]: Each list is a backward partitioning of the path.
    """
    repairs = []
    for p in ordered_partitions(path):
        rep = [(e[-1][1], e[0][0]) for e in p]  # Close each partition as (end_node, start_node)
        repairs.append(rep)
    return repairs


def repair_matrix_lower(logits, threshold=0.5, max_distance=10):
    """
    Repairs the lower triangle of the matrix by inserting the least costly (closest-to-1)
    backward arcs that close existing forward paths.

    Args:
        logits: Matrix of logits (torch.Tensor).
        threshold: Arc activation threshold.

    Returns:
        torch.Tensor: New repaired matrix with fixed backward arcs.
    """
    logits = logits.clone().detach()
    upper = torch.triu(logits, diagonal=1)
    paths = find_opening_paths(logits, threshold, max_distance=max_distance)

    for path in paths:
        partitions = opening_paths_partitions(path)
        partitions_weight = [sum(1 - logits[cell] for cell in partition) for partition in partitions]
        best_partition = partitions[torch.argmin(torch.tensor(partitions_weight))]
        for i, j in best_partition:
            logits[i, j] = 1.0

    return torch.tril(logits, diagonal=-1) + upper


def find_closing_paths(logits, threshold=0.5, max_distance=10):
    """
    Find all individual backward arcs (i.e., closing arcs) in the lower triangle.

    Args:
        logits: Matrix of logits.
        threshold: Activation threshold.

    Returns:
        List[Tuple[int, int]]: Closing arcs in form (j, i).
    """
    n = logits.size(0)
    return [(j, i) for i in range(n) for j in range(i + 1, min(i + max_distance, n)) if logits[j, i] > threshold]


def closing_path_partitions(path):
    """
    Given a closing arc (j, i), generate all possible forward arc partitions that would justify it.

    Args:
        path (Tuple[int, int]): A closing arc.

    Returns:
        List[List[Tuple[int, int]]]: Forward arc partitions that close into the given arc.
    """
    j, i = path
    length = j - i
    if length <= 0:
        return []

    result = []
    for p in ordered_partitions(list(range(length))):
        k, l = i, i
        part = []
        for block in p:
            k = k + len(block)
            part.append((l, k))
            l = l + len(block)
        result.append(part)

    return result


def repair_matrix_upper(logits, threshold=0.5, max_distance=10):
    """
    Repairs the upper triangle of the matrix by completing missing forward arcs
    required to justify existing backward arcs.

    Args:
        logits: Matrix of logits.
        threshold: Activation threshold.

    Returns:
        torch.Tensor: New repaired matrix with forward arcs completed.
    """
    logits = logits.clone().detach()
    lower = torch.tril(logits, diagonal=-1)
    paths = find_closing_paths(logits, threshold, max_distance=max_distance)

    for path in paths:
        partitions = closing_path_partitions(path)
        partitions_weight = [sum(1 - logits[cell] for cell in partition) for partition in partitions]
        if partitions_weight:
            best_partition = partitions[torch.argmin(torch.tensor(partitions_weight))]
            for i, j in best_partition:
                logits[i, j] = 1.0

    return lower + torch.triu(logits, diagonal=1)


def repair_logits_matrix(logits: torch.Tensor, threshold: float = 0.5, max_distance=10) -> torch.Tensor:
    """
    Wrapper function to repair a logits matrix by ensuring all forward paths are closed
    and all backward arcs are supported by valid forward chains.

    This performs:
    1. Lower triangle repair: Adds minimal backward arcs to close existing forward paths.
    2. Upper triangle repair: Adds minimal forward arcs to justify existing backward arcs.

    Args:
        logits (torch.Tensor): The model-predicted logits matrix (N x N).
        threshold (float): Activation threshold for considering arcs as present.

    Returns:
        torch.Tensor: A repaired logits matrix, ready to be used as a pseudo-label.
    """
    # First, repair backward arcs to ensure all open forward paths are closed
    repaired_upper = repair_matrix_upper(logits, threshold=threshold, max_distance=max_distance)

    # Then, repair forward arcs to justify all backward connections
    fully_repaired = repair_matrix_lower(repaired_upper, threshold=threshold, max_distance=max_distance)

    return fully_repaired


def compute_repair_loss(logits: torch.Tensor, threshold: float = 0.5, max_distance=10) -> torch.Tensor:
    """
    Computes a BCE loss between the model's predicted logits and a repaired version
    of those logits, where invalid structures (e.g., open loops) have been corrected.

    This loss penalizes the model for predicting structures that cannot form valid entities.

    Args:
        logits (torch.Tensor): Raw model logits of shape (N, N).
        threshold (float): Threshold used to binarize arcs during repair logic.

    Returns:
        torch.Tensor: Scalar BCE loss between original and repaired logits.
    """
    with torch.no_grad():
        # Step 1: Repair lower triangle (add closing backward arcs)
        repaired_lower = repair_matrix_lower(logits, threshold=threshold)

        # Step 2: Repair upper triangle (add missing forward arcs)
        repaired_logits = repair_matrix_upper(repaired_lower, threshold=threshold)

    # BCE loss between original logits and repaired logits (used as pseudo-labels)
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(logits, repaired_logits)

