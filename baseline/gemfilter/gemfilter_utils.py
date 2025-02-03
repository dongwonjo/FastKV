import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv

def standard_dis_index(data, queries, k, norm=1, pool=False, kernel_size=5, sum_over_heads=False):
    inner_product = torch.matmul(queries, data.transpose(-1, -2))
    inner_product = inner_product[:, :, 0, :]
    if sum_over_heads:
        inner_product = torch.sum(inner_product, dim=1, keepdim=True)
    if pool:
        inner_product = F.avg_pool1d(
            inner_product, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    top_k = torch.topk(inner_product, k, dim=-1)
    indices = top_k.indices
    distances = top_k.values
    if norm != 1:
        distances = distances / norm
    return distances, indices


def find_context(self, query_states, key_states, print_idx_dis=False):
    b, h, n, d = key_states.shape
    if self.indecies is None and self.layer_idx == self.select_layer_idx:
        assert b == 1
        key_states_repeat = repeat_kv(
            key_states, self.num_key_value_groups)
        query_last_states = query_states[:, :, -1:, :]
        _, indices = standard_dis_index(key_states_repeat, query_last_states, min(
            self.topk, n), pool=True, sum_over_heads=True)
        self.indecies = indices
        if print_idx_dis:
            print(self.layer_idx, torch.min(torch.abs(indices-62383)))
    return

def get_layer_context(model, tokenizer, input_ids, layer_idx, print_context=False):
    decoder_layer = model.model.layers[layer_idx]
    idx = decoder_layer.self_attn.indecies[0, 0, :]
    values, _ = torch.sort(idx)
    values = values.to('cuda:0')
    new_input_ids = input_ids.gather(0, values)
    if print_context:
        print(tokenizer.decode(new_input_ids))
    return new_input_ids.unsqueeze(0)

def reduce_layer(model, layer_idx):
    original_layers = model.model.layers
    model.model.layers = model.model.layers[:layer_idx+1]
    return model, original_layers

def recover_layer(model, layers):
    model.model.layers = layers
    return model

def set_topk(model, topk, mode='gemfilter'):
    decoder_layers = model.model.layers
    for i in range(len(decoder_layers)):
        if mode == 'gemfilter':
            decoder_layers[i].self_attn.topk = topk
        else:
            raise NotImplementedError
    return

def set_select_mode(model, mode):
    decoder_layers = model.model.layers
    for i in range(len(decoder_layers)):
        decoder_layers[i].self_attn.select_mode = mode
    return

def set_select_layer(model, select_layer_idx):
    if select_layer_idx is None:
        select_layer_idx = model.model.layers[0].self_attn.select_layer_idx
    else:
        decoder_layers = model.model.layers
        for i in range(len(decoder_layers)):
            decoder_layers[i].self_attn.select_layer_idx = select_layer_idx
    return select_layer_idx

@torch.no_grad()
def gemfilter_generate(model, tokenizer, pred_token_idx, past_key_values, max_gen_len):
    generated_ids = [pred_token_idx.item()]
    for _ in range(max_gen_len):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        if pred_token_idx == tokenizer.eos_token_id:
            break
        generated_ids.append(pred_token_idx.item())
    return generated_ids

@torch.no_grad()
def gemfilter_generate_selection(input_ids, attn_mask, model, tokenizer, max_gen_len=50, select_layer_idx=None, print_context=False):
    set_select_mode(model, True)
    select_layer_idx = set_select_layer(model, select_layer_idx)
    model, original_layers = reduce_layer(model, select_layer_idx)
    _ = model(input_ids, attention_mask=attn_mask)
    
    new_input_ids = get_layer_context(
        model, tokenizer, input_ids[0], select_layer_idx, print_context=print_context)
    model = recover_layer(model, original_layers)
    
    set_select_mode(model, False)
    outputs = model(new_input_ids, attention_mask=attn_mask)

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    output_ids = gemfilter_generate(model, tokenizer, pred_token_idx, past_key_values, max_gen_len=max_gen_len)
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return response

@torch.no_grad()
def gemfilter_generate_selection_prefill(input_ids, attn_mask, model, tokenizer, select_layer_idx=None, print_context=False):
    set_select_mode(model, True)
    select_layer_idx = set_select_layer(model, select_layer_idx)
    model, original_layers = reduce_layer(model, select_layer_idx)
    _ = model(input_ids, attention_mask=attn_mask)
    
    new_input_ids = get_layer_context(
        model, tokenizer, input_ids[0], select_layer_idx, print_context=print_context)
    model = recover_layer(model, original_layers)
    
    set_select_mode(model, False)
    outputs = model(new_input_ids, attention_mask=attn_mask)

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    return pred_token_idx, past_key_values
