from importlib.metadata import version
import warnings
import transformers
from baseline.fullkv.model_forward import llama_model_forward, mistral_model_forward

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

    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward