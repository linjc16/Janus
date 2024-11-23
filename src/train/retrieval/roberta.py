from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import datetime
import torch
import random
import pdb

import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fiqa', help='dataset name')
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--model_save_path', type=str, default='output/retrieval')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if not exists os.path.join(args.data_path, args.dataset), download and unzip
    if not os.path.exists(os.path.join(args.data_path, args.dataset)):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
        out_dir = args.data_path
        data_path = util.download_and_unzip(url, out_dir)
        # delete the zip file
        os.remove(os.path.join(out_dir, "{}.zip".format(args.dataset)))
    else:
        data_path = os.path.join(args.data_path, args.dataset)
    
    os.makedirs(args.model_save_path, exist_ok=True)

    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
    #### Please Note not all datasets contain a dev split, comment out the line if such the case
    # dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
    
    #### Provide any sentence-transformers or HF model
    model_name = "FacebookAI/roberta-large" 
    word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    #### Or provide pretrained sentence-transformer model
    # model = SentenceTransformer("msmarco-distilbert-base-v3")

    retriever = TrainRetriever(model=model, batch_size=16)

    #### Prepare training samples
    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    #### Training SBERT with cosine-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    #### training SBERT with dot-product
    # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)
    
    #### Prepare dev evaluator
    # ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
    
    #### If no dev set is present from above use dummy evaluator
    ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join(args.model_save_path, "{}-{}-{}".format(model_name, args.dataset, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(model_save_path, exist_ok=True)

    #### Configure Train params
    num_epochs = 5
    evaluation_steps = 5000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                    evaluator=ir_evaluator, 
                    epochs=num_epochs,
                    output_path=model_save_path,
                    warmup_steps=warmup_steps,
                    evaluation_steps=evaluation_steps,
                    use_amp=True)