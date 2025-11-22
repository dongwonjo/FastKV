import torch

import transformers
from transformers.models.llama.modeling_llama import  StaticCache

from baselines.fullkv.llama_model import llama_model_forward_general
from baselines.fullkv.mistral_model import mistral_model_forward_general

import json
import sys

def replace_llama(method):
    if method == "fastkv":
        from baselines.fastkv.llama_model import llama_model_forward_fastkv, llama_decoderlayer_forward_fastkv, LlamaFastKVAttention
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_fastkv
        transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = llama_decoderlayer_forward_fastkv
        transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFastKVAttention

    elif method == "streamingllm":
        from baselines.streamingllm.llama_model import llama_attn_forward_StreamingLLM, llama_flash_attn2_forward_StreamingLLM, llama_sdpa_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_general
    
    elif method == "h2o":
        from baselines.h2o.llama_model import llama_attn_forward_H2O, llama_flash_attn2_forward_H2O, llama_sdpa_attn_forward_H2O
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_H2O
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_general

    elif method == "snapkv":
        from baselines.snapkv.llama_model import llama_attn_forward_SnapKV, llama_flash_attn2_forward_SnapKV, llama_sdpa_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_general

    elif method == "gemfilter":
        from baselines.gemfilter.llama_model import LlamaGemFilterAttention
        transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaGemFilterAttention
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_general

    elif method == "pyramidinfer":
        from baselines.pyramidinfer import llama_model
        sys.modules["transformers.models.llama.modeling_llama"] = llama_model
    
    elif method == "fullkv":
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_general

    else:
        raise NotImplementedError(f"No method found for {method}")
    
    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama
        

def replace_mistral(method):
    if method == "fastkv":
        from baselines.fastkv.mistral_model import mistral_model_forward_fastkv, mistral_decoderlayer_forward_fastkv, MistralFastKVAttention
        transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_fastkv
        transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = mistral_decoderlayer_forward_fastkv
        transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES["flash_attention_2"] = MistralFastKVAttention
    
    elif method == "streamingllm":
        from baselines.streamingllm.mistral_model import mistral_attn_forward_StreamingLLM, mistral_flash_attn2_forward_StreamingLLM, mistral_sdpa_attn_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_general
        
    elif method == "h2o":
        from baselines.h2o.mistral_model import mistral_attn_forward_H2O, mistral_flash_attn2_forward_H2O, mistral_sdpa_attn_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_general

    elif method == "snapkv":
        from baselines.snapkv.mistral_model import mistral_attn_forward_SnapKV, mistral_flash_attn2_forward_SnapKV, mistral_sdpa_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_general

    elif method == "gemfilter":
        from baselines.gemfilter.mistral_model import MistralGemFilterAttention
        transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES["flash_attention_2"] = MistralGemFilterAttention

    elif method == "pyramidinfer":
        from baselines.pyramidinfer import mistral_model
        sys.modules["transformers.models.mistral.modeling_mistral"] = mistral_model
    
    elif method == "fullkv":
        transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_general

    else:
        raise NotImplementedError(f"No method found for {method}")

    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral

def set_model(model, args):
    if args.max_capacity_prompts != -1:
        max_capacity_prompts = args.max_capacity_prompts

    if args.method != "fullkv":
        if args.method in ["fullkv", "fastkv", "snapkv", "h2o"]:
            window_size = args.window_size
        elif args.method in ["streamingllm"]:
            window_size = max_capacity_prompts - 4
        elif args.method in ["gemfilter", "pyramidinfer"]:
            window_size = 1 # does not mean anything actually
            
        kernel_size = args.kernel_size
        pooling = args.pooling
        retain_rate = args.retain_rate


        layers = len(model.model.layers)
        # check if window_size is a list
        if not isinstance(window_size, list):
            window_size = [window_size] * layers
        if not isinstance(max_capacity_prompts, list):
            max_capacity_prompts = [max_capacity_prompts] * layers
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * layers
        if not isinstance(retain_rate, list):
            retain_rate = [retain_rate] * layers

        for i in range(layers):
            model.model.layers[i].self_attn.config.window_size = window_size[i]
            model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
            model.model.layers[i].self_attn.config.kernel_size = kernel_size[i]
            model.model.layers[i].self_attn.config.pooling = pooling
            model.model.layers[i].self_attn.config.merge = args.merge
            model.model.layers[i].self_attn.config.retain_rate = retain_rate[i]
            model.model.layers[i].self_attn.config.eviction_mode = args.eviction_mode

    
        # FastKV
        if args.method == "fastkv":
            from baselines.fastkv.utils import compress_fastkv
            args.window_size = window_size
            args.kernel_size = kernel_size
            compress_fastkv(model, args)

        elif args.method == "gemfilter":
            from baselines.gemfilter.utils import set_topk
            set_topk(model, args, mode='gemfilter')

        elif args.method  == "pyramidinfer":
            if "llama" in args.model_path.lower():
                if args.retain_rate == 0.35:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/llama31_8b_35%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.01
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.01
                elif args.retain_rate == 0.5:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/llama31_8b_50%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.3
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2
                elif args.retain_rate == 0.6:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/llama31_8b_60%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.7
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2
                else:
                    raise NotImplementedError(f"No config found for retain_rate={args.retain_rate}")
            elif "ministral" in args.model_path.lower():
                if args.retain_rate == 0.35:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/ministral_8b_35%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.01
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.01
                elif args.retain_rate == 0.6:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/ministral_8b_60%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.75
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2
                else:
                    raise NotImplementedError(f"No config found for retain_rate={args.retain_rate}")
            elif "nemo" in args.model_path.lower():
                if args.retain_rate == 0.6:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/nemo_12b_60%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.78
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2

            from baselines.pyramidinfer.utils import load_pyramid_config
            model = load_pyramid_config(model, pyramidinfer_config)


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


def prepare_inputs_for_generation_llama(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # if not isinstance(past_key_values, tuple):
        #     if len(past_key_values.key_cache) == 0:
        #         for layer in self.model.layers:
        #             layer.self_attn.kv_seq_len = 0
        
        ##### for 4.45 compatibility
        if past_key_values.get_seq_length() == 0:
            if isinstance(past_key_values.key_cache[0], list):
                for layer in self.model.layers:
                    layer.self_attn.kv_seq_len = 0
        
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        # import pdb;pdb.set_trace()

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    
def prepare_inputs_for_generation_mistral(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):

        # if not isinstance(past_key_values, tuple):
        #     if len(past_key_values.key_cache) == 0:
        #         for layer in self.model.layers:
        #             layer.self_attn.kv_seq_len = 0

        ##### for 4.45 compatibility
        if past_key_values.get_seq_length() == 0:
            if isinstance(past_key_values.key_cache[0], list):
                for layer in self.model.layers:
                    layer.self_attn.kv_seq_len = 0
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
