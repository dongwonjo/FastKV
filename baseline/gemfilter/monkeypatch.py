from importlib.metadata import version
import warnings
import transformers
from baseline.gemfilter.llama_hijack_4_43 import LlamaSelectAttention, llama_model_forward
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
from baseline.gemfilter.mistral_hijack_4_43 import MistralSelectAttention, mistral_model_forward
from transformers.models.mistral.modeling_mistral import MISTRAL_ATTENTION_CLASSES

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version


def replace_llama():
    transformers_version = check_version()
    version_list = ['4.43']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with FastKV. FastKV is tested with Transformers version {version_list}.")

    LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaSelectAttention
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward

def replace_mistral():
    transformers_version = check_version()
    version_list = ['4.43']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with FastKV. FastKV is tested with Transformers version {version_list}.")

    MISTRAL_ATTENTION_CLASSES['flash_attention_2'] = MistralSelectAttention
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward

