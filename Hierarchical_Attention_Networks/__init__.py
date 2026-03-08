"""
han_training — training pipeline for the Hierarchical Attention Network.

Modules
-------
config      TrainConfig dataclass (all hyper-parameters in one place)
tokenizer   Sentence splitting and word tokenisation
vocabulary  Vocabulary  (token ↔ id mapping, built from corpus)
dataset     HANDataset  + collate function
metrics     RunningMetrics  (loss / accuracy accumulator)
trainer     run_epoch, save_checkpoint, load_checkpoint
train       train()  main loop, parse_args(), smoke-test, CLI entry point
"""

from .config     import TrainConfig
from .tokenizer  import split_sentences, tokenise, tokenise_doc
from .vocabulary import Vocabulary, PAD_TOKEN, UNK_TOKEN
from .dataset    import HANDataset, han_collate, make_collate
from .metrics    import RunningMetrics
from .trainer    import run_epoch, save_checkpoint, load_checkpoint
from .train      import train, parse_args

__all__ = [
    "TrainConfig",
    "split_sentences", "tokenise", "tokenise_doc",
    "Vocabulary", "PAD_TOKEN", "UNK_TOKEN",
    "HANDataset", "han_collate", "make_collate",
    "RunningMetrics",
    "run_epoch", "save_checkpoint", "load_checkpoint",
    "train", "parse_args",
]
