import argparse
import os

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever

import random
import logging
import json
import pickle

import pdb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fiqa', choices=['fiqa', 'scifact'])
    parser.add_argument('--root_dir', type=str, default='data')
    args = parser.parse_args()

    
    data_path = os.path.join(args.root_dir, args.dataset)

    # output path, data path/statement
    output_path = os.path.join(data_path, 'statement')
    os.makedirs(output_path, exist_ok=True)

    def get_statement(split):
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        
        examples = []
        for query_id in qrels:
            for corpus_id, score in qrels[query_id].items():
                if score >= 1:

                    s1 = queries[query_id]
                    s2 = corpus[corpus_id].get("title") + " " + corpus[corpus_id].get("text")
                    # strip
                    s2 = s2.strip()
                    
                    answerKey = 'A'

                    ex_obj    = {"id": query_id + "_" + corpus_id, 
                                "question": {"stem": s1, "choices": [{'text': s2}]}, 
                                "answerKey": answerKey, 
                                }
                    examples.append(ex_obj)


        # shuffle
        random.seed(42)
        random.shuffle(examples)

        # save
        with open(os.path.join(output_path, f'{split}.statement.jsonl'), 'w') as fout:
            for dic in examples:
                print(json.dumps(dic), file=fout)

    for split in ['train', 'dev', 'test']:
        get_statement(split)

    

    def get_all_corpus_statement():
        corpus, _, _ = GenericDataLoader(data_path).load(split='test')

        examples = []
        for corpus_id in corpus:
                s2 = corpus[corpus_id].get("title") + " " + corpus[corpus_id].get("text")
                # strip
                s2 = s2.strip()
                
                answerKey = 'A'

                ex_obj    = {"id": corpus_id, 
                            "question": {"stem": s2, "choices": [{'text': ""}]}, 
                            "answerKey": answerKey, 
                            }
                examples.append(ex_obj)

        # save
        with open(os.path.join(output_path, f'corpus.statement.jsonl'), 'w') as fout:
            for dic in examples:
                print(json.dumps(dic), file=fout)

    
    get_all_corpus_statement()