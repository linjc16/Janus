#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
dt=`date '+%Y%m%d_%H%M%S'`


dataset="obqa"
model='intfloat/e5-mistral-7b-instruct'
model_prefix='encoder'
# model='mistralai/Mistral-7B-Instruct-v0.3'
# model_prefix='decoder'
shift
shift
args=$@


elr="5e-6"
dlr="1e-4"
bs=12
mbs=12
# bs=8
# mbs=8
n_epochs=5
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


k=5 #num of gnn layers
gnndim=256

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
for seed in 0 1 2; do
  python3 -u src/train/qa/mistral_dg_scores_only_text_embs.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed \
      --loss cross_entropy \
      --lr_schedule warmup_linear \
      --graph_text_embs_path data/${dataset}/graph/train.graph.embeddings.pt \
      -ebs $mbs --log_interval 10\
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/qa/text_only/${dataset}/train_mistraldg_scores_only_text_embs_${model_prefix}_${dataset}__enc-__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done