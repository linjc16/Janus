#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
dt=`date '+%Y%m%d_%H%M%S'`


dataset="scifact"
model='roberta-large'
shift
shift
args=$@


elr="1e-4"
dlr="1e-3"
bs=16
mbs=16
# bs=8
# mbs=8
n_epochs=15
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges
unfrz=0

k=3 #num of gnn layers
gnndim=512

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='/data/linjc/qagnn/saved_models/retrieval/qagnn'
mkdir -p $save_dir_pref
mkdir -p logs

###### Training ######
for seed in 0 1 2; do
  python3 -u src/train/retrieval/qagnn.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed \
      --all_corpus_statements_path data/${dataset}/statement/corpus.statement.jsonl \
      --all_corpus_adj_path data/${dataset}/corpus.graph.adj.pk \
      -ebs $mbs --log_interval 10 \
      --num_relation $num_relation \
      --fc_dim 512 \
      --fc_layer_num 2 \
      --unfreeze_epoch $unfrz \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/retrieval/qagnn/train_qagnn_${model_prefix}_${dataset}__enc-__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done