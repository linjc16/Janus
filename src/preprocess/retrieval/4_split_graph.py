import argparse
import os

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever

import random
import logging
import json
import pickle
from tqdm import tqdm

import pdb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='scifact', choices=['fiqa', 'scifact'])
    parser.add_argument('--root_dir', type=str, default='data')
    args = parser.parse_args()

    
    data_path = os.path.join(args.root_dir, args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fiqa', choices=['fiqa', 'scifact'])
    parser.add_argument('--root_dir', type=str, default='data')
    args = parser.parse_args()
    
    data_path = os.path.join(args.root_dir, args.dataset)
    
    statement_path = os.path.join(data_path, 'statement')

    output_path = os.path.join(data_path, 'graph')
    os.makedirs(output_path, exist_ok=True)

    # get graph
    corpus_graph_path = os.path.join(data_path, 'corpus.graph.adj.pk')
    query_graph_path = os.path.join(data_path, 'queries.graph.adj.pk')


    def get_graph_dict(graph_path):
        # load corpus graph
        with open(graph_path, 'rb') as f:
            graph_list = pickle.load(f)

        try:
            # get graph_dict, key is ['_id'], value is the list element
            graph_dict = {graph['_id']: graph for graph in graph_list}
        except:
            pdb.set_trace()

        return graph_dict

    corpus_graph_dict = get_graph_dict(corpus_graph_path)
    query_graph_dict = get_graph_dict(query_graph_path)
    

    # read statement
    for split in ['train', 'dev', 'test']:
        try:
            with open(os.path.join(statement_path, f'{split}.statement.jsonl'), 'r') as fin:
                statement_list = [json.loads(line) for line in fin]
        except:
            continue

        # get graph
        query_graph_split_list = []
        corpus_graph_split_list = []
        for statement in tqdm(statement_list):
            query_id, corpus_id = statement['id'].split('_')
            
            query_graph = query_graph_dict[query_id]
            corpus_graph = corpus_graph_dict[corpus_id]

            query_graph_split_list.append(query_graph)
            corpus_graph_split_list.append(corpus_graph)
        

        assert len(query_graph_split_list) == len(corpus_graph_split_list)

        # zip and save
        graph_split_list = list(zip(query_graph_split_list, corpus_graph_split_list))
        with open(os.path.join(output_path, f'{split}.graph.adj.pk'), 'wb') as f:
            pickle.dump(graph_split_list, f)
    