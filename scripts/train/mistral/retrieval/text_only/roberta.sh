
export CUDA_VISIBLE_DEVICES=4

DATASET=scifact
DATA_DIR=data

dt=`date '+%Y%m%d_%H%M%S'`

for seed in 1 2; do
    python src/train/retrieval/roberta.py \
        --dataset $DATASET \
        --data_path $DATA_DIR \
        --seed $seed \
    > logs/retrieval/text_only/train/roberta_large_${DATASET}_${dt}_${seed}.log
done