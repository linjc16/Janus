
export CUDA_VISIBLE_DEVICES=4

DATASET=scifact
DATA_DIR=data

dt=`date '+%Y%m%d_%H%M%S'`


python src/eval/retrieval/e5-mistral.py \
    --dataset $DATASET \
    --data_path $DATA_DIR \
> logs/retrieval/text_only/eval/e5-mistral_${DATASET}_${dt}_${seed}.txt