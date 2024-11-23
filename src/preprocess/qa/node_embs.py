import torch
from transformers import AutoTokenizer, AutoModel
import pdb
import os
from tqdm import tqdm
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2,6,7'


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



cache_dir = '/srv/local/data/jl254/cache'  
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)

concept_path = 'data/cpnet/concept.txt'

input_texts = []
with open(concept_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines:
    processed_line = line.strip().replace('_', ' ')
    input_texts.append(processed_line)


max_length = 80
batch_size = 64

# according to the batch_size, split the input_texts into batches
input_texts_batches = [input_texts[i:i + batch_size] for i in range(0, len(input_texts), batch_size)]


output_dict = {}
for idx, input_texts_batch in enumerate(tqdm(input_texts_batches)):
    batch_dict = tokenizer(input_texts_batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    embedding_dict = {batch_size * idx + i: embedding.cpu().detach().numpy() for i, embedding in enumerate(embeddings)}

    output_dict.update(embedding_dict)
    
    if idx % 100 == 0:
        with open(f'data/cpnet/concept_embeddings.npy', 'wb') as file:
            np.save(file, output_dict)


with open(f'data/cpnet/concept_embeddings.npy', 'wb') as file:
    np.save(file, output_dict)
