#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
dt=`date '+%Y%m%d_%H%M%S'`


dataset="webnlg"
model='roberta-large'
model_prefix='greaselm'
shift
shift
args=$@


elr="1e-4"
dlr="1e-4"
bs=32
mbs=32
sl=256
n_epochs=5
ent_emb='dbpedia'
num_relation=748 #(15 +2) * 2: originally 15, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


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
for seed in 0; do
  python3 -u src/train/pair_classification/greaselm.py --dataset $dataset \
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
      --save_dir ${save_dir_pref}/${dataset}/enc-roberta__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/pair_classification/no_contrastive/${model_prefix}_${dataset}__enc-roberta__k${k}__gnndim${gnndim}__bs${bs}__sl${sl}__unfrz${unfrz}__seed${seed}__${dt}.log.txt
done
