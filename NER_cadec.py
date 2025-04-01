import os
import json
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from EntityMatrixDataset import EntityMatrixDataset, collate_fn
from EntityMatrixPredictor import EntityMatrixPredictor
import torch
import torch.nn as nn

def compute_pos_weights(loader, device='cpu'):
    upper_ones = 0
    upper_zeros = 0
    lower_ones = 0
    lower_zeros = 0
    total_sentences = 0
    for batch in loader:
        tokens, target_matrix_batch, _ = batch
        input_ids = tokens['input_ids'].to(device)
        target_matrix_batch = target_matrix_batch.to(device)
        word_ids_batch = [tokens.word_ids(batch_index=i) for i in range(len(input_ids))]
        for i, matrix in enumerate(target_matrix_batch):
            word_ids = word_ids_batch[i]
            max_word_id = max([wid for wid in word_ids if wid is not None], default=-1)
            valid_len = max_word_id + 1
            if valid_len == 0:
                continue
            matrix = matrix[:valid_len, :valid_len]
            upper = torch.triu(matrix, diagonal=0)
            lower = torch.tril(matrix, diagonal=-1)
            upper_ones += (upper == 1).sum().item()
            upper_zeros += (upper == 0).sum().item()
            lower_ones += (lower == 1).sum().item()
            lower_zeros += (lower == 0).sum().item()
            total_sentences += 1
    print(f'Stats over {total_sentences} valid samples:')
    print(f'Upper Triangle: 1s={upper_ones}, 0s={upper_zeros}')
    print(f'Lower Triangle: 1s={lower_ones}, 0s={lower_zeros}')
    if upper_ones > 0:
        upper_pos_weight = upper_zeros / upper_ones
        print(f'Suggested upper pos_weight: {upper_pos_weight:.2f}')
    else:
        upper_pos_weight = None
        print('No 1s in upper triangle')
    if lower_ones > 0:
        lower_pos_weight = lower_zeros / lower_ones
        print(f'Suggested lower pos_weight: {lower_pos_weight:.2f}')
    else:
        lower_pos_weight = None
        print('No 1s in lower triangle')
    return (upper_pos_weight, lower_pos_weight)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_folder(file_path)
        os.rmdir(folder_path)
        print(f'Deleted folder: {folder_path}')
    else:
        print('Folder does not exist.')

def convert_cadec_to_matrix(dataset_dir, output_dir):
    dataset = load_from_disk(dataset_dir)
    os.makedirs(output_dir, exist_ok=True)
    dataset = dataset['train'].train_test_split(test_size=0.3, seed=42)
    test_valid = dataset['test'].train_test_split(test_size=0.5, seed=42)
    split_datasets = {'train': dataset['train'], 'test': test_valid['train'], 'validation': test_valid['test']}
    for split_name, split_data in split_datasets.items():
        save_matrices(output_dir, split_data, split_name)
    print(f'Formatted CADEC dataset saved to {output_dir} with 70/15/15 split.')

def save_matrices(output_dir, dataset_split, split_name):
    """
    Save entity matrices in the required format without modification.
    """
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for idx, example in enumerate(dataset_split):
        words = example['text'].split()
        entities = example['ade']
        entity_matrix = np.array(example['entity_matrix'])
        entry = {'words': words, 'entity_matrix': entity_matrix.tolist(), 'entities': entities}
        file_path = os.path.join(split_dir, f'sentence_{idx}.json')
        with open(file_path, 'w') as f:
            json.dump(entry, f)

def get_train_loader(dataset_dir, subset_size=None):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
    train_dir = f'{dataset_dir}/train'
    train_dataset = EntityMatrixDataset(train_dir, tokenizer)
    if subset_size is not None:
        train_dataset = Subset(train_dataset, list(range(subset_size)))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    return train_loader

def training_loop(dataset_dir, model=EntityMatrixPredictor(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epochs=3, pos_weight=15, verbose=False, subset_size=None):
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
            loss = loss_bce(predicted_matrix, target_matrix)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f'Batch loss {loss.item()}')
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        if verbose:
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
    return model

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

def evaluation_loop_sharp_path(model_path, dataset_dir):
    model = EntityMatrixPredictor(bert_model_name='bert-base-cased')
    model.load_state_dict(torch.load(model_path))
    return evaluation_loop_sharp(dataset_dir, model)

def evaluation_loop_sharp(dataset_dir, model):
    """
    Evaluates the model using extracted spans for precision, recall, and F1-score.

    Args:
        model_path (str): Path to the trained model.
    
    Returns:
        Precision, Recall, F1 Score based on span predictions.
    """
    bert_model_name = 'bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)
    eval_dir = f'{dataset_dir}/validation'
    eval_dataset = EntityMatrixDataset(eval_dir, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer), pin_memory=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    total_tp, total_fp, total_fn = (0, 0, 0)
    with torch.no_grad():
        for batch in eval_loader:
            tokens, _, spans = batch
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            word_ids = [tokens.word_ids(batch_index=i) for i in range(len(input_ids))]
            predicted_matrix = model(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids)
            binary_predictions = (predicted_matrix > 0.5).long()
            for i in range(len(input_ids)):
                max_word_idx = max([wid for wid in word_ids[i] if wid is not None], default=-1) + 1
                pred_matrix = binary_predictions[i, :max_word_idx, :max_word_idx]
                pred_spans = extract_spans_from_matrix(pred_matrix.cpu())
                true_spans = spans[i]
                tp, fp, fn = (0, 0, 0)
                for pred_span in pred_spans:
                    if pred_span in true_spans:
                        tp += 1
                    else:
                        fp += 1
                for true_span in true_spans:
                    if true_span not in pred_spans:
                        fn += 1
                total_tp += tp
                total_fp += fp
                total_fn += fn
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return (precision, recall, f1)

def evaluation_loop_path(model_path, dataset_dir):
    model = EntityMatrixPredictor(bert_model_name='bert-base-cased')
    model.load_state_dict(torch.load(model_path))
    return evaluation_loop(model, dataset_dir)

def evaluation_loop(dataset_dir, model):
    """
    Evaluates the model by directly comparing the predicted and target matrices.

    Args:
        model_path (str): Path to the trained model.
    
    Returns:
        Precision, Recall, F1 Score computed at the matrix level.
    """
    bert_model_name = 'bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)
    eval_dir = f'{dataset_dir}/validation'
    eval_dataset = EntityMatrixDataset(eval_dir, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer), pin_memory=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in eval_loader:
            tokens, target_matrices, _ = batch
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            word_ids = [tokens.word_ids(batch_index=i) for i in range(len(input_ids))]
            predicted_matrix = model(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids)
            binary_predictions = (predicted_matrix > 0.5).long()
            for i in range(len(input_ids)):
                max_word_idx = max([wid for wid in word_ids[i] if wid is not None], default=-1) + 1
                pred_matrix = binary_predictions[i, :max_word_idx, :max_word_idx]
                target_matrix = target_matrices[i, :max_word_idx, :max_word_idx]
                all_preds.extend(pred_matrix.cpu().numpy().flatten())
                all_targets.extend(target_matrix.cpu().numpy().flatten())
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    return (precision, recall, f1)

def visualize(dataset_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EntityMatrixPredictor().to(device)
    model.load_state_dict(torch.load(f'{model_path}_upper', map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
    file_path = f'{dataset_dir}/train/sentence_{random.randint(0, 600)}.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    words = data['words']
    entity_matrix = np.array(data['entity_matrix']).reshape(len(words), len(words))
    entities = data['entities']
    print('Words:')
    print(words)
    print('\nEntities:')
    print(entities)

    def find_entity_spans(words, entity_list):
        spans = []
        for entity in entity_list:
            entity_words = entity.split()
            for i in range(len(words) - len(entity_words) + 1):
                if words[i:i + len(entity_words)] == entity_words:
                    spans.append((i, i + len(entity_words) - 1))
                    break
        return spans
    spans = find_entity_spans(words, entities)

    def highlight_spans(words, spans):
        highlighted = words[:]
        for start, end in spans:
            for i in range(start, end + 1):
                highlighted[i] = f'[{words[i]}]'
        return highlighted
    highlighted_words = highlight_spans(words, spans)
    print('\nHighlighted Words:')
    print(' '.join(highlighted_words))
    encoding = tokenizer(words, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    word_ids = encoding.word_ids()
    with torch.no_grad():
        pred_matrix = model(input_ids=input_ids, attention_mask=attention_mask, word_ids=[word_ids]).squeeze(0).cpu().numpy()
    pred_matrix_binary = (pred_matrix > 0.5).astype(int)
    if np.all(pred_matrix_binary == 1) or np.all(pred_matrix_binary == 0):
        vmin, vmax = (entity_matrix.min(), entity_matrix.max())
    else:
        vmin, vmax = (0, 1)

    def plot_heatmap(matrix, title, words, cmap, vmin, vmax):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix[:len(words), :len(words)], annot=False, cbar=True, square=True, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(ticks=np.arange(len(words)) + 0.5, labels=words, rotation=90)
        plt.yticks(ticks=np.arange(len(words)) + 0.5, labels=words, rotation=0)
        plt.title(title)
        plt.show()
    plot_heatmap(entity_matrix, 'Ground Truth Entity Matrix', words, cmap='Blues', vmin=0, vmax=1)
    plot_heatmap(pred_matrix_binary, 'Predicted Entity Matrix', words, cmap='Reds', vmin=vmin, vmax=vmax)

def compute_metrics(pos_weight_values, num_runs, metrics_file, best_model_path):
    best_f1 = 0
    best_pos_weight = None
    with open(metrics_file, 'a') as f:
        f.write('\n---\n')
        f.write(f"### Experiment on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for pos_weight in pos_weight_values:
        total_precision, total_recall, total_f1 = (0, 0, 0)
        print(f'\nTesting pos_weight = {pos_weight}...')
        for _ in range(num_runs):
            model = training_loop(pos_weight=pos_weight, epochs=3, verbose=False)
            precision, recall, f1 = evaluation_loop(model)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        avg_precision = total_precision / num_runs
        avg_recall = total_recall / num_runs
        avg_f1 = total_f1 / num_runs
        print(f'pos_weight = {pos_weight} -> Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}')
        with open(metrics_file, 'a') as f:
            f.write(f'\n#### pos_weight = {pos_weight}\n')
            f.write(f'- **Precision:** {avg_precision:.4f}\n')
            f.write(f'- **Recall:** {avg_recall:.4f}\n')
            f.write(f'- **F1 Score:** {avg_f1:.4f}\n')
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_pos_weight = pos_weight
            torch.save(model.state_dict(), best_model_path)
    with open(metrics_file, 'a') as f:
        f.write('\n### Best Hyperparameter:\n')
        f.write(f'- **Best pos_weight:** {best_pos_weight}\n')
        f.write(f'- **Best F1 Score:** {best_f1:.4f}\n')
    print(f'\n✅ Best pos_weight found: {best_pos_weight} with F1 Score: {best_f1:.4f}')