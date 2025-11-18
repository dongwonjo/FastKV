import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaAttention,
    LlamaFlashAttention2
)
from transformers.utils import (
    logging,
    is_flash_attn_greater_or_equal_2_10,
)

from baselines.gemfilter.utils import find_context

logger = logging.get_logger(__name__)

def _flash_attention_forward(
    self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        )

    # if self.layer_idx == 0:
    #     import pdb; pdb.set_trace()

    return attn_output


class LlamaGemFilterAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.reset()
        self.topk = 1024
        self.rate = 0.25
        self.eviction_mode = "constant"
        self.select_layer_idx = 13
        self.select_mode = False

    def reset(self):
        self.indecies = None
        return

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        # will become mandatory in v4.45
        position_embeddings: Optional[Tuple[torch.Tensor,
                                            torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        # [GemFilter] update below
        if self.select_mode:
            self.reset()
            find_context(self, query_states, key_states)

        if not self.select_mode and past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        # [GemFilter] update above
        
        attn_output, attn_weights = self.flash_softmax(
            query_states, key_states, value_states, attention_mask, q_len, position_ids)
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def flash_softmax(self, query_states, key_states, value_states, attention_mask, q_len, position_ids):
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)
        
        attn_output = _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0.0,
            # sliding_window=getattr(self, "sliding_window", None),
            # use_top_left_mask=self._flash_attn_uses_top_left_mask,
            # is_causal=True,
        )

        if input_dtype == torch.float32:
            attn_output = attn_output.to(torch.float32)
        return attn_output, None
