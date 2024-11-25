import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertSdpaSelfAttention, 
    BertSelfOutput, 
    BertIntermediate, 
    BertOutput,
    BertEmbeddings,
    BertPooler
)
from module import LowRankAttention
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import functools
from util import device,preprocess_function,dataset_keys
import time
import numpy as np
import json
import os
import sys
sys.stdout=open('./eval.log','a',buffering=1)

time_record={
    BertForSequenceClassification.__name__:[],
    BertSdpaSelfAttention.__name__:[],
    BertSelfOutput.__name__:[],
    BertIntermediate.__name__:[],
    BertOutput.__name__:[],
    BertEmbeddings.__name__:[],
    BertPooler.__name__:[]
}

def add_time_warp(module,type):
    ori_forward=module.forward
    
    @functools.wraps(ori_forward)
    def forward_time(*args,**kwargs):
        time_start=time.perf_counter()
        output=ori_forward(*args,**kwargs)
        time_elapsed=time.perf_counter()-time_start
        time_record[type].append(time_elapsed*1000)
        return output
    module.forward=forward_time

def evaluate_model(model, tokenizer, dataset_name):
    print(f"\nEvaluating model on dataset {dataset_name}")

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

    
    print(f"Evaluating split {split}")
    eval_dataset = encoded_dataset[split].with_format("torch")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt")
    )
    
    model.eval()
    predictions = []
    labels = []
    for batch in tqdm(eval_dataloader):
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
        print(f"{dataset_name} ({split}) results: Pearson correlation: {metric_result['pearson']:.4f}, Spearman correlation: {metric_result['spearmanr']:.4f}")
    elif dataset_name in ['mrpc', 'qqp']:
        print(f"{dataset_name} ({split}) results: Accuracy: {metric_result['accuracy']:.4f}, F1 score: {metric_result['f1']:.4f}")
    elif dataset_name == 'cola':
        print(f"{dataset_name} ({split}) results: Matthews correlation: {metric_result['matthews_correlation']:.4f}")
    else:
        print(f"{dataset_name} ({split}) results: Accuracy: {metric_result['accuracy']:.4f}")

if __name__=='__main__':
    # dataset_list = [
    #     'sst2', 'qnli', 'rte', 'mrpc', 'qqp', 'cola', 'mnli', 'stsb'
    # ]
    dataset_list = [
        'sst2', 'qnli', 'rte', 'mrpc', 'qqp', 'cola', 'mnli', 'stsb'
    ]
    tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
    if not os.path.exists('./time.json'):
        time_result={}
    else:
        time_result=json.load(open('./time.json','r'))
    for dataset_name in dataset_list:
        time_record={k:[] for k,v in time_record.items()}
        # ori model
        #model_name=f'./models/{dataset_name}'
        #model=AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        # prone model
        model=torch.load(f'./prone/{dataset_name}.pt')
        # svd model
        # model=torch.load(f'./svd/{dataset_name}.pt')
        # prone retrain model
        # model=torch.load(f'./prone_retrain/{dataset_name}.pt')
        # svd retrain model
        # model=torch.load(f'./svd_retrain/{dataset_name}.pt')
        add_time_warp(model,BertForSequenceClassification.__name__)
        for name, module in model.named_modules():
            if isinstance(module, (BertSdpaSelfAttention,BertEmbeddings,BertPooler)):
                add_time_warp(module,type(module).__name__)
            elif isinstance(module, (BertSelfOutput,BertIntermediate,BertOutput)):
                add_time_warp(module,type(module).__name__)
            elif isinstance(module, (LowRankAttention)):
                add_time_warp(module,BertSdpaSelfAttention.__name__)
        evaluate_model(model, tokenizer, dataset_name)
        time_record={k:np.mean(v) for k,v in time_record.items()}
        time_result[dataset_name]=time_record
    json.dump(time_result,open('./time.json','w'),indent=1)