from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fiqa', choices=['fiqa', 'scifact'], help='dataset name')
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--model_path', type=str, default='output/retrieval')
    args = parser.parse_args()

    # if not exists os.path.join(args.data_path, args.dataset), download and unzip
    if not os.path.exists(os.path.join(args.data_path, args.dataset)):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
        out_dir = args.data_path
        data_path = util.download_and_unzip(url, out_dir)
        # delete the zip file
        os.remove(os.path.join(out_dir, "{}.zip".format(args.dataset)))
    else:
        data_path = os.path.join(args.data_path, args.dataset)
    

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


    cache_dir = '/srv/local/data/jl254/cache'
    model_path = args.model_path
    # model_path = 'src/train/retrieval/output/FacebookAI/roberta-large-v1-fiqa'
    model = DRES(models.SentenceBERT(model_path, cache_dir=cache_dir), batch_size=6)
    

    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    logging.info("Evaluation Results with k = %s" % retriever.k_values)
    logging.info("NDCG@k: %s" % ndcg)
    logging.info("MAP@k: %s" % _map)
    logging.info("Recall@k: %s" % recall)
    logging.info("Precision@k: %s" % precision)

    #### Print top-k documents retrieved ####
    top_k = 10

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info("Query : %s\n" % queries[query_id])

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))