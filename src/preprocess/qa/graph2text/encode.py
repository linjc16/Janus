import pickle
import pdb
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
import os
import json

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

import argparse

# use cuda 5
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def read_examples(input_file):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label
    
    with open(input_file, "r", encoding="utf-8") as f:
        examples = []
        for line in f.readlines():
            json_dic = json.loads(line)
            label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
            contexts = json_dic["question"]["stem"]
            if "para" in json_dic:
                contexts = json_dic["para"] + " " + contexts
            if "fact1" in json_dic:
                contexts = json_dic["fact1"] + " " + contexts
            examples.append(
                InputExample(
                    example_id=json_dic["id"],
                    contexts=[contexts] * len(json_dic["question"]["choices"]),
                    question="",
                    endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                    label=label
                ))
    return examples



relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
]

def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


def create_matcher_patterns(cpnet_vocab_path, output_path, debug=False):
    cpnet_vocab = load_cpnet_vocab(cpnet_vocab_path)

def read_adj_concept_pairs(adj_pk_path):
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)
    return adj_concept_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='dev', choices=['train', 'dev', 'test'], help='split')
    parser.add_argument('--dataset', default='csqa', choices=['csqa', 'obqa'], help='dataset')
    parser.add_argument('--max_seq_length', default=4096, type=int, help='max sequence length')
    parser.add_argument('--model_name', type=str, choices=[
        'e5-mistral', 'e5-base-v2', 'e5-small-v2', 'e5-large-v2', 'llama-2-7b', 'llama-2-13b', 
        'llama-3.2-1b', 'llama-3.2-3b', 'llama-3-8b'], default='e5-mistral')
    args = parser.parse_args()
    
    cpnet_vocab = load_cpnet_vocab("data/cpnet/concept.txt")
    
    adj_concept_pairs = read_adj_concept_pairs(f"data/{args.dataset}/graph/{args.split}.graph.adj.pk")
    n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices

    cache_dir = '/srv/local/data/jl254/cache'

    statement_path = f'data/{args.dataset}/statement/{args.split}.statement.jsonl'
    
    examples = read_examples(statement_path)

    num_choices = len(examples[0].endings)

    max_seq_length = args.max_seq_length
    
    model_name_dict = {
        'e5-mistral': 'intfloat/e5-mistral-7b-instruct',
        'e5-base-v2': 'intfloat/e5-base-v2',
        'e5-small-v2': 'intfloat/e5-small-v2',
        'e5-large-v2': 'intfloat/e5-large-v2',
        'llama-2-7b': 'NousResearch/Llama-2-7b-hf',
        'llama-2-13b': 'NousResearch/Llama-2-13b-hf',
        'llama-3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama-3.2-1b': 'meta-llama/Llama-3.2-1B',
        'llama-3.2-3b': 'meta-llama/Llama-3.2-3B',
    }

    model_name = model_name_dict[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id


    triplet_text_list_dict = defaultdict(list)

    for _idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
        
        adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
        
        #Load adj
        ij = np.array(adj.row) #(num_matrix_entries, ), where each entry is coordinate
        k = np.array(adj.col)  #(num_matrix_entries, ), where each entry is coordinate
        n_node = adj.shape[1]
        half_n_rel = adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node
        
        triples_list = []
        
        # for idx in range(len(i)):
        #     # relation is 
        for idx_g in range(len(i)):
            triples_list.append((cpnet_vocab[concepts[j[idx_g]]], relation_text[i[idx_g]], cpnet_vocab[concepts[k[idx_g]]]))
        
        triples_list_text = "; ".join([f"{h} {r} {t}" for h, r, t in triples_list])

        # if word count exceeds args.max_seq_length, only keep (args.max_seq_length - 128) words
        if len(triples_list_text.split()) > max_seq_length - 128:
            triples_list_text = " ".join(triples_list_text.split()[:max_seq_length - 128])
        

        triplet_text_list_dict[_idx // num_choices].append(triples_list_text)
    
    generated_embeddings = []
    input_text = []
    # convert examples
    for ex_index, example in enumerate(tqdm(examples)):
        endings = example.endings

        for ending_idx, (context, ending, triplet_text) in enumerate(zip(example.contexts, example.endings, triplet_text_list_dict[ex_index])):
            
            query = ""
            # query = "Given a multi-choice question and choice candidates, retrieve the correct answer. You can refer to the input external knowledge if needed. "
            query += '\n\nExternal knowledge: ' + triplet_text + '\n\n'
            query += context + " "
            query += example.question + " " if example.question != "" else ""
            query = query.strip()
            inputs = query + " " + ending
            
            input_text.append(inputs)
    

    batch_size = 2
    # split the input_text into batches
    input_text_batches = [input_text[i:i + batch_size] for i in range(0, len(input_text), batch_size)]
    
    # tokenize the input_text
    for inputs in tqdm(input_text_batches, desc='tokenizing'):
        
        batch_dict = tokenizer(inputs, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt').to(model.device)     

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            generated_embeddings.append(embeddings.cpu().detach())
            
    
    
    generated_embeddings = torch.cat(generated_embeddings, dim=0)
    
    # # save the embeddings to .pt
    # with open(f'data/{args.dataset}/graph/train.graph.embeddings_filter.pt', 'wb') as file:
    #     torch.save(generated_embeddings, file)

    # save the embeddings to .pt
    with open(f'data/{args.dataset}/graph/{args.split}.graph.embeddings_cat_query_candidate_nofilter_{args.model_name}.pt', 'wb') as file:
        torch.save(generated_embeddings, file)