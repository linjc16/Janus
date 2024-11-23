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


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='csqa', choices=['csqa', 'obqa'], help='dataset')
    parser.add_argument('--max_seq_length', default=4096, type=int, help='max sequence length')
    parser.add_argument('--model_name', type=str, choices=['e5-mistral', 'sfr', 'e5-base-v2', 'e5-small-v2', 'e5-large-v2'], default='e5-small-v2')
    args = parser.parse_args()
    
    cpnet_vocab = load_cpnet_vocab("data/cpnet/concept.txt")

    cache_dir = '/srv/local/data/jl254/cache'

    model_name_dict = {
        'e5-mistral': 'intfloat/e5-mistral-7b-instruct',
        'sfr': 'Salesforce/SFR-Embedding-2_R',
        'e5-base-v2': 'intfloat/e5-base-v2',
        'e5-small-v2': 'intfloat/e5-small-v2',
        'e5-large-v2': 'intfloat/e5-large-v2',
    }

    model_name = model_name_dict[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
    

    batch_size = 1024
    # split the cpnet_vocab into batches
    input_concept_batches = [cpnet_vocab[i:i + batch_size] for i in range(0, len(cpnet_vocab), batch_size)]

    # encode cpnet_vocab, save embeddings to {name}.ent.npy
    generated_embeddings = []
    for input in tqdm(input_concept_batches):
        batch_dict = tokenizer(input, max_length=args.max_seq_length, padding=True, truncation=True, return_tensors='pt').to(model.device)     

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            generated_embeddings.append(embeddings.cpu().detach())
    
    
    emb = torch.cat(generated_embeddings, dim=0).numpy()
    with open(f'data/cpnet/{args.model_name}.ent.npy', 'wb') as file:
        np.save(file, emb)