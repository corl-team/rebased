from src.config_pyr import TrainConfig, ModuleConfig


def update_sequence_mixer(config: TrainConfig):
    if "attention" in config.model.sequence_mixer.name or "rwkv" in config.model.sequence_mixer.name:
        config.model.num_heads = 1
    else:
        config.model.num_heads = 12
    
    MIXERS = get_mixers(config)
    config.model.sequence_mixer = ModuleConfig(**MIXERS[config.model.sequence_mixer.name])
    config.model.max_position_embeddings = config.data.input_seq_len
    
    if 'mamba' in config.model.sequence_mixer.name:
        config.model.block_type = "MambaBlock"
    else:
        config.model.block_type = "TransformerBlock"
    
    # only for main fig exps
    if config.data.input_seq_len == 128:
        config.data.builder.kwargs["num_kv_pairs"] = 16
    else:
        config.data.builder.kwargs["num_kv_pairs"] = config.data.input_seq_len // 4
    
    if (config.data.input_seq_len >= 512) and (config.data.batch_size > 64):
        '''
        seq_len -> max_bs
        1024 -> 256
        2048 -> 128
        4096 -> 64
        8192 -> 32
        '''
        max_bs = 256 // (config.data.input_seq_len // 512)
        if max_bs < config.data.batch_size:
            gradient_accumulation_steps = config.data.batch_size // max_bs
            config.gradient_accumulation_steps = gradient_accumulation_steps
            config.data.batch_size = max_bs
            print(f"Setting bs to {max_bs} with {gradient_accumulation_steps} gradient accumulation steps")

    return config


def get_mixers(config: TrainConfig) -> dict:
    return {
        "attention": dict(
            name="src.mixers.attention.MHA",
            kwargs={"dropout": 0.1, **config.model.sequence_mixer.kwargs},
        ),
        "conv_attention": dict(
            name="src.mixers.hybrid.Hybrid",
            kwargs={
                "configs": [
                    dict(
                        name="src.mixers.base_conv.BaseConv",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            # pass a list of kernel sizes for each of four layers
                            "kernel_size": 3,
                        },
                    ),
                    dict(
                        name="src.mixers.attention.MHA",
                        kwargs={"dropout": 0.1, **config.model.sequence_mixer.kwargs},
                    ),
                ]
            },
        ),
        "based": dict(
            name="src.mixers.hybrid.Hybrid",
            kwargs={
                "configs": [
                    dict(
                        name="src.mixers.base_conv.BaseConv",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            # pass a list of kernel sizes for each of four layers
                            "kernel_size": 3,
                        },
                    ),
                    dict(
                        name="src.mixers.based.Based",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            "feature_dim": 16,
                            "num_key_value_heads": config.model.num_heads,
                            "num_heads": config.model.num_heads,
                            "feature_name": "taylor_exp",
                            "qk_norm": False,
                            # **config.model.sequence_mixer.kwargs
                        },
                    ),
                ]
            },
        ),
        "rebased": dict(
            name="src.mixers.hybrid.Hybrid",
            kwargs={
                "configs": [
                    dict(
                        name="src.mixers.base_conv.BaseConv",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            # pass a list of kernel sizes for each of four layers
                            "kernel_size": 3,
                        },
                    ),
                    dict(
                        name="src.mixers.rebased.ReBased",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            "feature_dim": 16,
                            "num_key_value_heads": config.model.num_heads,
                            "num_heads": config.model.num_heads,
                            "feature_name": "taylor_exp",
                            "use_beta": config.model.use_beta,
                            "use_gamma": config.model.use_gamma,
                            "normalize": config.model.normalize,
                            # **config.model.sequence_mixer.kwargs
                        },
                    ),
                ]
            },
        ),
        "mamba": dict(
            name="src.mixers.mamba.Mamba",
            kwargs={
                "d_state": 16,
                **config.model.sequence_mixer.kwargs
            }
        ),
        "rwkv": dict(
            name="src.mixers.rwkv.RWKVTimeMixer",
            kwargs={
                "l_max": config.data.input_seq_len,
                "n_layer": 2,
                **config.model.sequence_mixer.kwargs
            },
        ),
        "conv_rwkv": dict(
            name="src.mixers.hybrid.Hybrid",
            kwargs={
                "configs": [
                    dict(
                        name="src.mixers.base_conv.BaseConv",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            # pass a list of kernel sizes for each of four layers
                            "kernel_size": 3,
                        },
                    ),
                    dict(
                        name="src.mixers.rwkv.RWKVTimeMixer",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            "n_layer": 2,
                            **config.model.sequence_mixer.kwargs
                        },
                    ),
                ]
            },
        ),
        "double_based": dict(
            name="src.mixers.hybrid.Hybrid",
            kwargs={
                "configs": [
                    dict(
                        name="src.mixers.based.Based",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            "feature_dim": 16,
                            "num_key_value_heads": config.model.num_heads,
                            "num_heads": config.model.num_heads,
                            "feature_name": "taylor_exp",
                            "qk_norm": False,
                            # **config.model.sequence_mixer.kwargs
                        },
                    ),
                    dict(
                        name="src.mixers.based.Based",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            "feature_dim": 16,
                            "num_key_value_heads": config.model.num_heads,
                            "num_heads": config.model.num_heads,
                            "feature_name": "taylor_exp",
                            "qk_norm": False,
                            # **config.model.sequence_mixer.kwargs
                        },
                    ),
                ]
            },
        ),
        "double_rebased": dict(
            name="src.mixers.hybrid.Hybrid",
            kwargs={
                "configs": [
                    dict(
                        name="src.mixers.rebased.ReBased",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            "feature_dim": 16,
                            "num_key_value_heads": config.model.num_heads,
                            "num_heads": config.model.num_heads,
                            "feature_name": "taylor_exp",
                            "use_beta": config.model.use_beta,
                            "use_gamma": config.model.use_gamma,
                            "normalize": config.model.normalize,
                            # **config.model.sequence_mixer.kwargs
                        },
                    ),
                    dict(
                        name="src.mixers.rebased.ReBased",
                        kwargs={
                            "l_max": config.data.input_seq_len,
                            "feature_dim": 16,
                            "num_key_value_heads": config.model.num_heads,
                            "num_heads": config.model.num_heads,
                            "feature_name": "taylor_exp",
                            "use_beta": config.model.use_beta,
                            "use_gamma": config.model.use_gamma,
                            "normalize": config.model.normalize,
                            # **config.model.sequence_mixer.kwargs
                        },
                    ),
                ]
            },
        ),
    }
