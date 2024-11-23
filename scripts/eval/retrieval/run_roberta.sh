DATASET=scifact
DATA_DIR=data
MODEL_NAME=roberta-large
# MODEL_PATH=output/retrieval/FacebookAI/$MODEL_NAME-v1-$DATASET
# MODEL_PATH=output/retrieval/FacebookAI/roberta-large-scifact-2024-10-01_08-57-20
MODEL_PATH=output/retrieval/FacebookAI/roberta-large-scifact-2024-10-01_08-59-00
# MODEL_PATH=output/retrieval/FacebookAI/roberta-large-scifact-2024-10-01_09-02-01
dt=`date '+%Y%m%d_%H%M%S'`

CUDA_VISIBLE_DEVICES=0 python src/eval/retrieval/roberta.py \
    --dataset $DATASET \
    --data_path $DATA_DIR \
    --model_path $MODEL_PATH \
> logs/retrieval/text_only/eval/roberta-large_${DATASET}_${dt}.txt