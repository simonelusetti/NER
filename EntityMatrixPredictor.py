import torch
import torch.nn as nn
from transformers import BertModel

class EntityMatrixPredictor(nn.Module):
    def __init__(self, bert_model_name='bert-base-cased', hidden_dim=768, num_heads=4, dropout=0.1):
        super(EntityMatrixPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.mlp_forward = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.v_forward = nn.Parameter(torch.randn(hidden_dim))
        self.mlp_backward = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.v_backward = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, input_ids, attention_mask, word_ids):
        batch_size, _ = input_ids.shape
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output.last_hidden_state
        max_words = max([max([wid for wid in word_id if wid is not None], default=-1) + 1 for word_id in word_ids])
        word_embeddings = torch.zeros((batch_size, max_words, token_embeddings.shape[-1]), device=token_embeddings.device)
        for i in range(batch_size):
            word_counts = torch.zeros((max_words, 1), device=token_embeddings.device)
            for token_idx, word_idx in enumerate(word_ids[i]):
                if word_idx is not None:
                    word_embeddings[i, word_idx] += token_embeddings[i, token_idx]
                    word_counts[word_idx] += 1
            word_embeddings[i] /= word_counts.clamp(min=1)
        i_emb = word_embeddings.unsqueeze(2).expand(-1, -1, max_words, -1)
        j_emb = word_embeddings.unsqueeze(1).expand(-1, max_words, -1, -1)
        pair_matrix = torch.cat((i_emb, j_emb), dim=-1)
        logits_forward = torch.triu(torch.matmul(self.mlp_forward(pair_matrix), self.v_forward))
        logits_backward = torch.tril(torch.matmul(self.mlp_backward(pair_matrix), self.v_backward), diagonal=-1)
        return logits_forward + logits_backward
