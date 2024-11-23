#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
dt=`date '+%Y%m%d_%H%M%S'`


dataset="medqa_usmle"
model='intfloat/e5-mistral-7b-instruct'
shift
shift
args=$@


elr="1e-5"
dlr="1e-4"
bs=8
mbs=8
sl=256
n_epochs=15
ent_emb='ddb'
num_relation=34 #(15 +2) * 2: originally 15, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


k=5 #num of gnn layers
gnndim=256
unfrz=100000


echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='/data/linjc/qagnn/saved_models'
mkdir -p $save_dir_pref
mkdir -p logs

###### Training ######
for seed in 1 2 3 4 5; do
  python3 -u src/train/mistral_dg_scores_nognn.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs -sl $sl --fp16 true --seed $seed \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 10 --unfreeze_epoch $unfrz \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --ent_emb ${ent_emb} \
      --save_dir ${save_dir_pref}/${dataset}/enc-sapbert__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/nognn/${dataset}/train_mistraldg_scores_nognn_${dataset}__enc-sapbert__k${k}__gnndim${gnndim}__bs${bs}__sl${sl}__unfrz${unfrz}__seed${seed}__${dt}.log.txt
done