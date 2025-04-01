from torch.utils.data import Dataset
import os
import json
import torch

class EntityMatrixDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_seq_length=128):
        """
        Initialize the dataset by loading all JSON files from the directory.
        
        Args:
            data_dir: Path to the directory containing JSON files.
            tokenizer: Tokenizer to encode tokens.
            max_seq_length: Maximum sequence length for padding/truncation.
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(data_dir, file_name), 'r') as f:
                    entry = json.load(f)
                    entry['entity_matrix'] = entry['entity_matrix']
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the words and single-word entity vector for the given index.
        
        Returns:
            - words: List of words in the sentence.
            - single_word_labels: Tensor with 1s for single-word entities, 0s otherwise.
        """
        entry = self.data[idx]
        words = entry['words']
        entities = entry['entities']
        entity_matrix = torch.tensor(entry['entity_matrix'], dtype=torch.float32)
        return (words, entity_matrix, entities)

def collate_fn(batch, tokenizer, max_seq_length=128):
    """
    Collate function to pad words and entity matrices within a batch.

    Args:
        batch: List of tuples (words, entity_matrix).
        tokenizer: Tokenizer to encode and pad words.
        max_seq_length: Maximum sequence length for padding.

    Returns:
        - Encoded tokenized inputs (input_ids, attention_mask, word_ids).
        - Padded entity matrices tensor.
    """
    words_batch, matrices_batch, entities = zip(*batch)
    encoded = tokenizer(list(words_batch), is_split_into_words=True, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
    batch_size = len(batch)
    padded_seq_len = max_seq_length
    padded_matrices = torch.zeros((batch_size, padded_seq_len, padded_seq_len), dtype=torch.float32)
    for i, (matrix, words) in enumerate(zip(matrices_batch, words_batch)):
        size = len(words)
        matrix = matrix.reshape((size, size))
        seq_len = min(matrix.shape[0], padded_seq_len)
        padded_matrices[i, :seq_len, :seq_len] = matrix[:seq_len, :seq_len]
    return (encoded, padded_matrices, entities)