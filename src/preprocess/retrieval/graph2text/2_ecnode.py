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
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fiqa', choices=['fiqa', 'scifact'], help='dataset')
    parser.add_argument('--max_seq_length', default=4096, type=int, help='max sequence length')
    args = parser.parse_args()
    
    cache_dir = '/srv/local/data/jl254/cache'
    
    statement_path = f'data/{args.dataset}/statement/train.statement.jsonl'
    
    examples = read_examples(statement_path)

    num_choices = len(examples[0].endings)

    max_seq_length = args.max_seq_length


    model_name = 'intfloat/e5-mistral-7b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
    
    def save_embeddings(frame):
        with open(f'data/{args.dataset}/graph/train.{frame}.graph2text.json', 'r') as f:
            triplet_text_list_dict = json.load(f)
        
        generated_embeddings = []
        input_text = []
        # convert examples
        for ex_index, example in enumerate(tqdm(examples)):
            endings = example.endings
            
            for ending_idx, (context, ending, triplet_text) in enumerate(zip(example.contexts, example.endings, triplet_text_list_dict[example.example_id])):
                
                query = 'External Knowledge: ' + triplet_text + '\n\n'
                sentence = context if frame == 'queries' else ending
                query += 'Sentences: ' + sentence + " "
                query = query.strip()
                inputs = query

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
        

        # save the embeddings to .pt
        with open(f'data/{args.dataset}/graph/train.{frame}.graph.embeds_cat.pt', 'wb') as file:
            torch.save(generated_embeddings, file)
    
    for frame in ['queries', 'corpus']:
        save_embeddings(frame)