import torch

def loop_reward(pred_matrix):
    pred_binary = (pred_matrix > 0.5).int()

    split_matrices = split_lower_triangular_ones_torch(pred_binary)
    if not split_matrices:
        return torch.tensor(0.0, device=pred_matrix.device)

    loop_flags = [find_loops_torch(m) for m in split_matrices]
    nodes_with_loops = torch.stack(loop_flags).any(dim=0)
    nodes_with_edges = nodes_with_arcs_torch(pred_binary)

    # XNOR equivalent: ~(A ^ B)
    valid_nodes = ~(nodes_with_loops ^ (~nodes_with_edges))
    num_valid = valid_nodes.sum().float()

    return num_valid / pred_matrix.shape[0]

def reachability_torch(m, k):
    return torch.diag(torch.matrix_power(m, k))

def find_loops_torch(adj_matrix):
    n = adj_matrix.shape[0]
    loop_diag = [reachability_torch(adj_matrix, i) for i in range(1, n + 1)]
    return torch.stack(loop_diag).sum(dim=0).bool()  # shape: (n,)

def split_lower_triangular_ones_torch(matrix):
    upper_triangle = torch.triu(matrix)
    lower_positions = (torch.tril(matrix, diagonal=-1) == 1).nonzero(as_tuple=False)

    matrices = []
    for pos in lower_positions:
        i, j = pos
        new_matrix = upper_triangle.clone()
        new_matrix[i, j] = 1
        matrices.append(new_matrix)

    return matrices

def nodes_with_arcs_torch(adj_matrix):
    return (adj_matrix.sum(dim=0) > 0) | (adj_matrix.sum(dim=1) > 0)

def extract_spans_from_matrix(matrix):
    """
    Extracts and merges entity spans by collapsing linked words in the **upper triangular** part of the entity matrix.

    Args:
        matrix (torch.Tensor): Binary entity matrix (size: max_words x max_words).

    Returns:
        merged_spans (set of tuples): Extracted entity spans in (start, end) format.
    """
    max_words = matrix.shape[0]
    spans = []

    # **Step 1: Extract Raw Spans from Upper Triangle**
    for i in range(max_words):
        if matrix[i, i] == 1:
                spans.append([i, i])

    start = -1
    for i in range(max_words-1):
        if matrix[i,i+1] == 1:
            if start == -1:
                start = i
        elif start != -1:
            spans.append([start,i])
            start = -1
    if start != -1:
         spans.append([start,max_words-1])

    return spans  # Convert to set for unique values
