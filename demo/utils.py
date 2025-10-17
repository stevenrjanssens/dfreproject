import os

import psutil
import torch


def _log_memory(stage: str):
    """Simple helper to log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"[MEMORY] {stage}: GPU allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")
    else:
        process = psutil.Process(os.getpid())
        ram = process.memory_info().rss / 1024**2  # MB
        print(f"[MEMORY] {stage}: CPU RAM={ram:.1f}MB")
