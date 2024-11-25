# DRONE
DRONE: Data-aware Low-rank Compression for Large NLP Models

https://proceedings.neurips.cc/paper_files/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf

## Step 1
Run train.py with retrain=False, this script will finetune bert-base-uncased model on 8 datasets of GLUE , including 'sst2', 'qnli', 'rte', 'mrpc', 'qqp', 'cola', 'mnli' and 'stsb'

## Step 2
Run eval.py to evaluate the effect of finetuned bertmodel on GLUE dataset. 

This process also calculates the average elapsed time of the Attention and Linear layers in Bert and saves it in a json file.

The execution of the drone algorithm relies on this elapsed time data.

## Step 3
Run drone.py.

The full DRONE algorithm is reimplemented in this script, which will calculate the optimal low rank approximation of BertSdpaSelfAttention, BertSelfOutput, BertSelfOutput, and BertOutput, the four layers of the Bert model, and then replaces shese layers with the LowRankLinear,LowRankAttention classes implemented in the module.py file. 

The low rank approximated model will be stored in the ./drone subdirectory.

It also records to json all the compressed layers and their corresponding ranks, which are needed to implement traditional svd methods.

## Step 4

Run svd.py.

It will read the saved all_compressed_ranks.json file, and compress the corresponding layers in the model using the SVD method with the same rank as drone, and eventually save the new model in ./svd subdirectory.

## Step 4

Run train.py with retrain=True, method="drone" or "svd".
This script will retrain the low rank approximated models on 10% training data from the original training dataset.
Retrained models will be save to ./svd_retrain and ./drone_retrain subdirectories respectively.

## Step 5
Run eval.py for all the models to obtain the final evaluation result. 