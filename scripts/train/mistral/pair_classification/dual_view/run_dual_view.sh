export CUDA_VISIBLE_DEVICES=2
dt=`date '+%Y%m%d_%H%M%S'`


dataset="webnlg"
model='intfloat/e5-mistral-7b-instruct'
model_prefix='encoder'
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
for seed in 0 1 2; do
  python3 -u src/train/pair_classification/mistral_dual_view.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed \
      --loss cross_entropy \
      --graph_text_embs_path data/${dataset}/graph/train.graph.embeddings_cat_query_candidate_filter.pt \
      -ebs $mbs --log_interval 10\
      --num_relation $num_relation \
      --ent_emb ${ent_emb} \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/pair_classification/dual_view/${model_prefix}_${dataset}__enc-__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done