import transformers

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
            elif "mistral" in args.model_path.lower() or "ministral" in args.model_path.lower():
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

            from baselines.pyramidinfer.utils import load_pyramid_config
            model = load_pyramid_config(model, pyramidinfer_config)