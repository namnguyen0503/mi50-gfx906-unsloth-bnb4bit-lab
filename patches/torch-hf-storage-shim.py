"""Compatibility shim for torch 2.7 + transformers 5.x layouts.

Place this content at:
`torch/distributed/checkpoint/hf_storage.py`

It re-exports the renamed Hugging Face checkpoint storage symbols from
`torch.distributed.checkpoint._hf_storage`.
"""

from torch.distributed.checkpoint._hf_storage import (
    _HuggingFaceStorageReader as HuggingFaceStorageReader,
    _HuggingFaceStorageWriter as HuggingFaceStorageWriter,
)

__all__ = ["HuggingFaceStorageReader", "HuggingFaceStorageWriter"]
