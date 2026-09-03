"""Training entry points for the three surrogates."""

from .train_downscaler import train_downscaler
from .train_fno import train_fno
from .train_mgn import train_mgn

__all__ = ["train_downscaler", "train_fno", "train_mgn"]
