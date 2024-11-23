from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random
import argparse
import torch

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fiqa', help='dataset name')
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=6)
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
    

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    INSTRUCTION = {
        "scifact": "Given a scientific claim, retrieve documents that support or refute the claim.",
        "fiqa": " Given a financial question, retrieve user replies that best answer the question.",
    }

    # for each query, add the instruction before the query
    for query_id in queries:
        queries[query_id] = INSTRUCTION[args.dataset] + "\n" + queries[query_id]
    
    #### Dense Retrieval using ANCE #### 
    # https://www.sbert.net/docs/pretrained-models/msmarco-v3.html
    # MSMARCO Dev Passage Retrieval ANCE(FirstP) 600K model from ANCE.
    # The ANCE model was fine-tuned using dot-product (dot) function.

    cache_dir = '/srv/local/data/jl254/cache'
    model = DRES(models.SentenceBERT("intfloat/e5-mistral-7b-instruct", cache_dir=cache_dir), batch_size=args.batch_size)
    
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