import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from transformers import AutoModel, AutoTokenizer, BertTokenizer,DataCollatorWithPadding,AutoModelForSequenceClassification

from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertSdpaSelfAttention,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertEmbeddings,
    BertPooler
)
from module import LowRankLinear,LowRankAttention
from util import device,preprocess_function,dataset_keys
from tqdm.auto import tqdm
from datasets import load_dataset
import evaluate
from util import low_rank_approximation, low_rank_approximation_attn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
#torch.backends.cuda.preferred_linalg_library("magma") 
import sys
sys.stdout=open('./compress.log','a',buffering=1)

n_layer = 12
tolerant = 2  # loss increase ratio
dataset_times=json.load(open('./time_ori_gpu.json','r'))
steps={
        BertSdpaSelfAttention:16,
        BertSelfOutput:96,
        BertIntermediate:96,
        BertOutput:96
}
max_ranks={
        BertSdpaSelfAttention:32,
        BertSelfOutput:384,
        BertIntermediate:384,
        BertOutput:384
}

def get_parent_module(model, target_module):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is target_module:
                return module, child_name

def get_tolerances(dataset_name):
    times=dataset_times[dataset_name]
    
    time_attn = times[BertSdpaSelfAttention.__name__]
    time_self_output = times[BertSelfOutput.__name__]
    time_intermediate = times[BertIntermediate.__name__]
    time_output = times[BertOutput.__name__]

    minimal_time = min(time_attn, time_self_output, time_intermediate, time_output)
    multiplier = (time_attn + time_self_output + time_intermediate + time_output) / minimal_time

    basic_tolerance = np.exp(np.log(tolerant) / multiplier)

    tol_attn = np.exp(np.log(basic_tolerance ** (time_attn / minimal_time)) / n_layer)
    tol_self_output = np.exp(np.log(basic_tolerance ** (time_self_output / minimal_time)) / n_layer)
    tol_intermediate = np.exp(np.log(basic_tolerance ** (time_intermediate / minimal_time)) / n_layer)
    tol_output = np.exp(np.log(basic_tolerance ** (time_output / minimal_time)) / n_layer)
    
    tols={
        BertSdpaSelfAttention:tol_attn,
        BertSelfOutput:tol_self_output,
        BertIntermediate:tol_intermediate,
        BertOutput:tol_output
    }
    return tols

def collect_activations(model, dataloader, layer):
    activations = []
    def hook_fn(module, input, output):
        activations.append(input[0].detach())
        
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(next(model.parameters()).device)
            model(inputs)

    handle.remove()
    X = torch.cat(activations, dim=0)
    if X.ndim > 2:
        X = X.permute(2, 0, 1, *range(3, X.ndim))
        X = X.reshape(X.shape[0], -1)
    else:
        X = X.T
    return X

def compress_attn_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss, increment):
    rank = initial_rank

    Q = layer.query.weight.data.clone()
    K = layer.key.weight.data.clone()
    V = layer.value.weight.data.clone()
    bias_q = layer.query.bias.clone()
    bias_k = layer.key.bias.clone()
    bias_v = layer.value.bias.clone()
    X = collect_activations(model, dataloader, layer.query)

    parent, name = get_parent_module(model, layer)
    original_layer = getattr(parent, name)

    while True:
        
        U_Q,V_Q,U_K,V_K,M_U,M_V,U_V,V_V= low_rank_approximation_attn(Q,K,V,X,rank,bias_q=bias_q,bias_k=bias_k)

        low_rank_layer = LowRankAttention(U_Q.T,V_Q.T,U_K.T,V_K.T,U_V.T,V_V.T,bias_q,bias_k,bias_v,M_U,M_V)

        setattr(parent, name, low_rank_layer)
        
        new_loss = evaluate_model_loss(model, dataloader)
        
        if new_loss < allowed_loss:
            del X
            torch.cuda.empty_cache()
            return rank
        else:
            rank += increment
        if rank>max_rank: 
            del X
            torch.cuda.empty_cache()
            setattr(parent, name, original_layer)
            return False
        torch.cuda.empty_cache()

def compress_linear_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss, increment):
    rank = initial_rank
    original_weight = layer.dense.weight.data.clone()
    X = collect_activations(model, dataloader, layer)

    while True:
        U,V,_ = low_rank_approximation(original_weight, X, rank)
        layer.dense.weight.data = U@V
        
        new_loss = evaluate_model_loss(model, dataloader)
        
        if new_loss < allowed_loss:
            del X
            torch.cuda.empty_cache()
            setattr(layer,"dense",LowRankLinear(U.T,V.T,layer.dense.bias.data.clone()))
            return rank
        else:
            layer.dense.weight.data = original_weight
            rank += increment
        if rank>max_rank: 
            del X
            torch.cuda.empty_cache()
            return False
        torch.cuda.empty_cache()

def compress_layer(model, dataloader, layer, layer_name, initial_rank, max_rank, allowed_loss, compressed_ranks):
    if isinstance(layer, BertSdpaSelfAttention): #attention layer
        optimal_rank = compress_attn_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss,initial_rank)
    else:
        optimal_rank = compress_linear_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss,initial_rank)
    
    if optimal_rank:
        print("Layer compressed with optimal rank ", optimal_rank)
        compressed_ranks[layer_name] = optimal_rank
        return True
    else:
        print("Can not find optimal rank within the loss tolerance")
        return False

def overall_low_rank_approximation(model, dataloader, tols, compressed_ranks):
    layer_names = [name for name, _ in model.named_modules()]

    original_loss = evaluate_model_loss(model, dataloader)
    allowed_loss = original_loss
       
    for layer_name in layer_names:
        try:
            layer = dict(model.named_modules())[layer_name]
        except Exception as e:
            continue
        if isinstance(layer, (BertSdpaSelfAttention,BertSelfOutput,BertIntermediate,BertOutput)):
            tol = tols[type(layer)]
            initial_rank = steps[type(layer)]
            max_rank = max_ranks[type(layer)]
        else :
            continue

        allowed_loss = original_loss * tol

        # Compress the layer
        print("------------------")
        print("Layer Name:", layer_name)
        print("allowed_loss", allowed_loss)
        result = compress_layer(model, dataloader, layer, layer_name, initial_rank, max_rank, allowed_loss, compressed_ranks)
        
        if result:
            new_loss = evaluate_model_loss(model, dataloader)
            print("Original_loss:", original_loss)
            print("New_loss:", new_loss)
        else:
            print("Layer not compressed")

    return model

def evaluate_model_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_model_accuracy(model, dataloader,dataset_name):
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
                if prediction.size==1: prediction=[prediction]
                label = labels_batch.squeeze().cpu().numpy()
                if label.size==1: label=[label]
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

def compress(model, tokenizer, dataset_name, all_compressed_ranks):
    tols=get_tolerances(dataset_name)
    
    print(f"\n Compress model on dataset {dataset_name}")

    dataset = load_dataset('glue', dataset_name)

    sentence1_key, sentence2_key = dataset_keys[dataset_name]
    encoded_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, sentence1_key, sentence2_key),
        batched=True,
        remove_columns=[col for col in dataset['train'].column_names],
        num_proc=2,
        load_from_cache_file=True
        )

    # mnli has two validation set
    if dataset_name == 'mnli':
        split = 'validation_matched'
    else:
        split = 'validation'
    
    eval_dataset = encoded_dataset[split].with_format("torch")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        #collate_fn=DataCollatorWithPadding(tokenizer, padding=True,return_tensors="pt")
    )

    evaluate_model_accuracy(model, eval_dataloader,dataset_name)
    
    compressed_ranks = {}
    compressed_model = overall_low_rank_approximation(model, eval_dataloader, tols, compressed_ranks)

    torch.save(compressed_model, f'prone/{dataset_name}.pt')

    # Store the compressed ranks in the global dictionary under the dataset name
    all_compressed_ranks[dataset_name] = compressed_ranks

    print(f"Evaluating compressed model")
    
    evaluate_model_accuracy(compressed_model, eval_dataloader,dataset_name)
    

if __name__=='__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    # dataset_list = [
    #     'sst2', 'qnli', 'rte', 'mrpc', 'qqp', 'cola', 'mnli', 'stsb'
    # ]
    dataset_list = [
        'stsb'
    ]
    tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
    if not os.path.exists('all_compressed_ranks.json'):
        all_compressed_ranks = {}
    else:
        all_compressed_ranks=json.load(open('all_compressed_ranks.json','r'))
    
    for dataset_name in dataset_list:
        if dataset_name in all_compressed_ranks.keys(): continue
        model_name=f'./models/{dataset_name}'
        model=AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        compress(model, tokenizer, dataset_name, all_compressed_ranks)
        with open('all_compressed_ranks.json', 'w') as f:
            json.dump(all_compressed_ranks, f, indent=4)
