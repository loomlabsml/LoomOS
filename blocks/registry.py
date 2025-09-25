# Registry

from .adapters.hf_model_adapter import HFModelAdapter

registry = {
    "hf-model": HFModelAdapter
}