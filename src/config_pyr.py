from dataclasses import dataclass, field
from src.utils import import_from_str
from functools import partial
from dataclasses import asdict
import os
from typing import List


@dataclass
class FunctionConfig:
    name: str
    kwargs: dict = field(default_factory=dict)

    def instantiate(self):
        return partial(import_from_str(self.name), **self.kwargs)


@dataclass
class ModuleConfig:
    name: str
    kwargs: dict = field(default_factory=dict)

    def instantiate(self, **kwargs):
        return import_from_str(self.name)(**kwargs, **self.kwargs)


@dataclass
class DataConfig:
    builder: FunctionConfig = None
    seed: int = 42

    num_train_examples: int = 100_000
    num_test_examples: int = 3_000
    input_seq_len: int = 64
    vocab_size: int = 8_192
    batch_size: int = 32
    

    cache_dir: str = "/app/cache"  # if os.path.exists("/app/cache") else "cache",)
    caching: bool = True
    force_cache: bool = False

    def dict(self):
        return asdict(self)


@dataclass
class ModelConfig:
    sequence_mixer: ModuleConfig = None
    state_mixer: ModuleConfig = None

    affine_norm: bool = True
    pre_norm: bool = True
    use_gamma: bool = True
    use_beta: bool = True
    normalize: bool = True

    d_model: int = 128
    n_layers: int = 2
    num_heads: int = 1
    max_position_embeddings: int = 64
    learnable_word_embeddings: bool = True
    vocab_size: int = 8_192

    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    drop_path: float = 0.0
    layer_norm_epsilon: float = 1e-5
    pad_vocab_size_multiple: int = 1
    gradient_checkpointing: bool = False

    block_type: str = "TransformerBlock"
    log_scores: bool = False


@dataclass
class LoggerConfig:
    project_name: str = None
    entity: str = None
    do_log_ckpt: bool = False
    do_log_gamma_beta: bool = False


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    tqdm: bool = False

    max_epochs: int = 100
    sequence_mixer: str = None

    # stop training once this metric reaches the threshold
    # set metric to None to disable early stopping
    early_stopping_metric: str = "valid/accuracy"
    early_stopping_threshold: float = 0.99

    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 123
    gradient_accumulation_steps: int = 1

    launch_id: str = None
    sweep_id: str = None
    run_id: str = "default"
    tags: List[str] = None
    _wandb: dict = field(default_factory=dict)

    def model_dump(self):
        return asdict(self)

    def print(self):
        return print(asdict(self))
