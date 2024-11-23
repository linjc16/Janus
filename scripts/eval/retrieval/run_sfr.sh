
export CUDA_VISIBLE_DEVICES=4

# DATASET=scifact
DATASET=fiqa
DATA_DIR=data

dt=`date '+%Y%m%d_%H%M%S'`


python src/eval/retrieval/sfr.py \
    --dataset $DATASET \
    --data_path $DATA_DIR \
> logs/retrieval/text_only/eval/sfr_${DATASET}_${dt}_${seed}.txt