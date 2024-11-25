import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from transformers import AutoModel, AutoTokenizer, BertTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertSelfAttention,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertEmbeddings,
    BertPooler
)
from module import LowRankLinear, LowRankAttention
from util import device, preprocess_function, dataset_keys
from tqdm.auto import tqdm
from datasets import load_dataset
import evaluate
from util import low_rank_approximation, low_rank_approximation_attn, low_rank_approximation_SVD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json

n_layer = 12

def compress_attn_layer_by_optimal_rank_svd(layer, rank):
    Q = layer.query.weight.data.clone()
    K = layer.key.weight.data.clone()
    V = layer.value.weight.data.clone()
    bias_q = layer.query.bias.clone()
    bias_k = layer.key.bias.clone()
    bias_v = layer.value.bias.clone()
    
    UQ, VQ, _ = low_rank_approximation_SVD(Q, 0, rank)
    UK, VK, _ = low_rank_approximation_SVD(K, 0, rank)
    UV, VV, _ = low_rank_approximation_SVD(V, 0, rank)

    setattr(layer, "query", LowRankLinear(UQ.T, VQ.T, bias_q))
    setattr(layer, "key", LowRankLinear(UK.T, VK.T, bias_k))
    setattr(layer, "value", LowRankLinear(UV.T, VV.T, bias_v))

def compress_linear_layer_by_optimal_rank_svd(layer, rank):
    original_weight = layer.dense.weight.data.clone()
    U, V, _ = low_rank_approximation_SVD(original_weight, 0, rank)
    setattr(layer, "dense", LowRankLinear(U.T, V.T, layer.dense.bias.data.clone()))

def compress_layer(layer, rank):
    if isinstance(layer, BertSelfAttention):  # Attention layer
        compress_attn_layer_by_optimal_rank_svd(layer, rank)
    elif isinstance(layer, (BertSelfOutput, BertIntermediate, BertOutput)):
        compress_linear_layer_by_optimal_rank_svd(layer, rank)
    else:
        print(f"Layer {layer} is not of a compressible type.")
        return
    print("Layer compressed with optimal rank ", rank)

def overall_low_rank_approximation(model, compressed_ranks):
    for layer_name, rank in compressed_ranks.items():
        try:
            layer = dict(model.named_modules())[layer_name]
        except Exception as e:
            print(f"Layer {layer_name} not found in the model.")
            continue
        # Compress the layer
        compress_layer(layer, rank)
        print(f"Compressed layer {layer_name} to rank {rank}.")
    return model

def evaluate_model_accuracy(model, dataloader, dataset_name):
    print(f"Evaluating ")
    model.eval()
    predictions = []
    labels = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            labels_batch = batch['labels']
            outputs = model(**batch)
            logits = outputs.logits
            if dataset_name == 'stsb':
                # regression
                prediction = logits.squeeze().cpu().numpy()
                if prediction.size == 1:
                    prediction = [prediction]
                label = labels_batch.squeeze().cpu().numpy()
                if label.size == 1:
                    label = [label]
            else:
                prediction = torch.argmax(logits, dim=-1).cpu().numpy()
                label = labels_batch.cpu().numpy()
            predictions.extend(prediction)
            labels.extend(label)

    # metrics
    metric = evaluate.load('glue', dataset_name)
    metric_result = metric.compute(predictions=predictions, references=labels)
    if dataset_name == 'stsb':
        print(f"{dataset_name} results: Pearson correlation: {metric_result['pearson']:.4f}, Spearman correlation: {metric_result['spearmanr']:.4f}")
    elif dataset_name in ['mrpc', 'qqp']:
        print(f"{dataset_name} results: Accuracy: {metric_result['accuracy']:.4f}, F1 score: {metric_result['f1']:.4f}")
    elif dataset_name == 'cola':
        print(f"{dataset_name} results: Matthews correlation: {metric_result['matthews_correlation']:.4f}")
    else:
        print(f"{dataset_name} results: Accuracy: {metric_result['accuracy']:.4f}")

def compress_svd(model, tokenizer, dataset_name, compressed_ranks):
    print(f"\n Compress model on dataset {dataset_name}")

    dataset = load_dataset('glue', dataset_name)
    sentence1_key, sentence2_key = dataset_keys[dataset_name]
    encoded_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, sentence1_key, sentence2_key),
        batched=True,
        remove_columns=[col for col in dataset['train'].column_names],
        num_proc=2
    )

    # mnli has two validation sets
    if dataset_name == 'mnli':
        split = 'validation_matched'
    else:
        split = 'validation'

    eval_dataset = encoded_dataset[split].with_format("torch")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        # collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt")
    )

    # Evaluate model accuracy before compression
    evaluate_model_accuracy(model, eval_dataloader, dataset_name)

    # Compress the model using the provided ranks
    compressed_model = overall_low_rank_approximation(model, compressed_ranks)

    torch.save(compressed_model, f'svd/{dataset_name}.pt')

    print(f"Evaluating compressed model")

    # Evaluate model accuracy after compression
    evaluate_model_accuracy(compressed_model, eval_dataloader, dataset_name)

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    dataset_list = [
        'sst2', 'qnli', 'rte', 'mrpc', 'qqp', 'cola', 'mnli', 'stsb'
    ]
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    all_compressed_ranks = json.load(open('all_compressed_ranks.json', 'r'))

    for dataset_name in dataset_list:
        if os.path.exists(f'svd/{dataset_name}.pt'): continue
        compressed_ranks = all_compressed_ranks[dataset_name]
        model_name = f'./models/{dataset_name}'
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        compress_svd(model, tokenizer, dataset_name, compressed_ranks)
