from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from util import device,preprocess_function,compute_metrics,dataset_keys
import sys
sys.stdout=open('./train.log','a',buffering=1)

def train_model(model_name, dataset_name):
    print(f"\nTraining model {model_name} on dataset {dataset_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset('glue', dataset_name)

    # Get number of labels or set regression task
    if dataset_name == 'stsb':
        num_labels = 1  # Regression task
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, problem_type='regression')
    else:
        num_labels = dataset['train'].features['label'].num_classes
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    sentence1_key, sentence2_key = dataset_keys[dataset_name]
    encoded_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, sentence1_key, sentence2_key),
        batched=True,num_proc=8,
        remove_columns=[col for col in dataset['train'].column_names if col not in ['label']]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/{dataset_name}',
        overwrite_output_dir=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_steps=1,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='matthews_correlation' if dataset_name=='cola' else 'pearson' if dataset_name == 'stsb' else 'accuracy',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation_matched'] if dataset_name=='mnli' else encoded_dataset['validation'],
        data_collator=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, dataset_name),
    )

    trainer.train()
    trainer.save_model(f"./models/{dataset_name}")

if __name__=='__main__':
    model_name = 'bert-base-uncased'
    # dataset_list = [
    #     'sst2', 'qnli', 'rte', 'mrpc', 'qqp', 'cola', 'mnli', 'stsb'
    # ]
    dataset_list = [
        'mnli',
    ]

    # Fine-tune and evaluate the model on each dataset
    for dataset_name in dataset_list:
        train_model(model_name, dataset_name)
