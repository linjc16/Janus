import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, BertModel, BertConfig
from utils.layers import *
from utils.data_utils import get_gpt_token_num
from torch import Tensor

MODEL_NAME_TO_CLASS = {}

#Add SapBERT configuration
model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'

model_name = 'roberta-large'
MODEL_NAME_TO_CLASS[model_name] = 'roberta'

model_name = 'intfloat/e5-mistral-7b-instruct'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'openai-community/gpt2'
MODEL_NAME_TO_CLASS[model_name] = 'gpt'

model_name = 'intfloat/e5-mistral-7b-instruct'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'Salesforce/SFR-Embedding-2_R'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'intfloat/e5-large-v2'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'intfloat/e5-small-v2'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'intfloat/e5-base-v2'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'NousResearch/Llama-2-13b-hf'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'NousResearch/Llama-2-7b-hf'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'meta-llama/Llama-3.2-1B'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'meta-llama/Llama-3.2-3B'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

model_name = 'meta-llama/Meta-Llama-3-8B'
MODEL_NAME_TO_CLASS[model_name] = 'mistral'

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class LSTMTextEncoder(nn.Module):
    pool_layer_classes = {'mean': MeanPoolLayer, 'max': MaxPoolLayer}

    def __init__(self, vocab_size=1, emb_size=300, hidden_size=300, output_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, pretrained_emb_or_path=None, freeze_emb=True,
                 pool_function='max', output_hidden_states=False):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_hidden_states = output_hidden_states
        assert not bidirectional or hidden_size % 2 == 0

        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                pretrained_emb_or_path = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb = nn.Embedding.from_pretrained(pretrained_emb_or_path, freeze=freeze_emb)
            emb_size = emb.weight.size(1)
        else:
            emb = nn.Embedding(vocab_size, emb_size)
        self.emb = EmbeddingDropout(emb, emb_p)
        self.rnns = nn.ModuleList([nn.LSTM(emb_size if l == 0 else hidden_size,
                                           (hidden_size if l != num_layers else output_size) // (2 if bidirectional else 1),
                                           1, bidirectional=bidirectional, batch_first=True) for l in range(num_layers)])
        self.pooler = self.pool_layer_classes[pool_function]()

        self.input_dropout = nn.Dropout(input_p)
        self.hidden_dropout = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        assert (lengths > 0).all()
        batch_size, seq_len = inputs.size()
        hidden_states = self.input_dropout(self.emb(inputs))
        all_hidden_states = [hidden_states]
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropout)):
            hidden_states = pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
            hidden_states, _ = rnn(hidden_states)
            hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=seq_len)
            all_hidden_states.append(hidden_states)
            if l != self.num_layers - 1:
                hidden_states = hid_dp(hidden_states)
        pooled = self.pooler(all_hidden_states[-1], lengths)
        assert len(all_hidden_states) == self.num_layers + 1
        outputs = (all_hidden_states[-1], pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs


class TextEncoder(nn.Module):
    # valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in ('bert', 'roberta', 'albert', 'mistral')

        if self.model_type in ('lstm',):
            self.module = LSTMTextEncoder(**kwargs, output_hidden_states=True)
            self.sent_dim = self.module.output_size
        elif self.model_type in ('mistral'):
            model_class = AutoModel
            self.module = model_class.from_pretrained(model_name, cache_dir='/srv/local/data/jl254/cache', device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
            self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size
            
            # from peft import LoraConfig
            # lora_config = LoraConfig(
            #     r=16,
            #     lora_alpha=16,
            #     lora_dropout=0.05,
            #     target_modules=[
            #         'q_proj',
            #         'k_proj',
            #         'v_proj',
            #         'o_proj',
            #         'gate_proj',
            #         'up_proj',
            #         'down_proj'
            #     ]
            # )
            # self.module.add_adapter(lora_config, adapter_name='lora')
            
        else:
            model_class = AutoModel
            self.module = model_class.from_pretrained(model_name, output_hidden_states=True)
            if from_checkpoint is not None:
                self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
            if self.model_type in ('gpt',):
                self.module.resize_token_embeddings(get_gpt_token_num())
            self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        layer_id: only works for non-LSTM encoders
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''

        if self.model_type in ('lstm',):  # lstm
            input_ids, lengths = inputs
            outputs = self.module(input_ids, lengths)
        elif self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.module(input_ids)
        elif self.model_type in ('mistral',):
            batch_dict = inputs[0]
            outputs = self.module(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            return embeddings, None
        
        else:  # bert / xlnet / roberta
            try:
                input_ids, attention_mask, token_type_ids, output_mask = inputs
                outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            except:
                input_ids = input_ids.view(-1, input_ids.size(-1))
                attention_mask = attention_mask.view(-1, attention_mask.size(-1))
                token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
                output_mask = output_mask.view(-1, output_mask.size(-1))
                
                outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]
        
        if self.model_type in ('lstm',):
            sent_vecs = outputs[1]
        elif self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        elif self.model_type in ('albert',):
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = hidden_states[:, 0]
        elif self.model_type in ('mistral',):
            raise NotImplementedError()
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = self.module.pooler(hidden_states)
        return sent_vecs, all_hidden_states


def run_test():
    encoder = TextEncoder('lstm', vocab_size=100, emb_size=100, hidden_size=200, num_layers=4)
    input_ids = torch.randint(0, 100, (30, 70))
    lenghts = torch.randint(1, 70, (30,))
    outputs = encoder(input_ids, lenghts)
    assert outputs[0].size() == (30, 200)
    assert len(outputs[1]) == 4 + 1
    assert all([x.size() == (30, 70, 100 if l == 0 else 200) for l, x in enumerate(outputs[1])])
    print('all tests are passed')
