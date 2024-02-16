from pathlib import Path

import wandb
from torch.nn import Module
import torch

from src.config_pyr import LoggerConfig, TrainConfig


class WandbLogger:
    def __init__(self, config: TrainConfig):
        if config.logger.project_name is None or config.logger.entity is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return
        self.no_logger = False
        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name,
            tags=config.tags,
        )
        self.run.mark_preempting()
        self.cache_dir = config.data.cache_dir
        self.model_name = config.model.sequence_mixer.name
        self.do_log_ckpt = config.logger.do_log_ckpt
        self.do_log_gamma_beta = config.logger.do_log_gamma_beta

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(self, model: Module):
        if self.no_logger:
            return
        wandb.watch(model)

    def log_ckpt(self, model: Module, tag: str):
        if self.do_log_ckpt:
            ckpt_path = f"{self.cache_dir}/ckpt_{tag}.pth"
            cktp_name = f"{self.run.id}_{self.model_name}_{tag}"
            torch.save(model.state_dict(), ckpt_path)
            print("Saving checkpoint: {} ...".format(ckpt_path))
            artifact = wandb.Artifact(name=cktp_name, type="model")
            artifact.add_file(local_path=ckpt_path)
            wandb.log_artifact(artifact)
    
    def log_gamma_beta(self, model: Module, global_step_idx: int):
        if not self.do_log_gamma_beta:
            return
        
        gamma_beta = {
            "gamma_q": model.backbone.layers[1].sequence_mixer.mixer.ln_q.weight,
            "gamma_k": model.backbone.layers[1].sequence_mixer.mixer.ln_k.weight,
            "beta_q": model.backbone.layers[1].sequence_mixer.mixer.ln_q.bias,
            "beta_k": model.backbone.layers[1].sequence_mixer.mixer.ln_k.bias
        }
        wandb.log({f"{k}_mean": v.mean() for k, v in gamma_beta.items()})
        
        if global_step_idx <= 200 and global_step_idx % 4 != 0:
            return
        if global_step_idx > 200 and global_step_idx % 100 != 0:
            return
                
        tensor_path = f"{self.cache_dir}/gamma_beta_{global_step_idx}.pth"
        artifact_name = f"{self.run.id}_gamma_beta"
        torch.save(gamma_beta, tensor_path)
        # print("Saving checkpoint: {} ...".format(tensor_path))
        artifact = wandb.Artifact(name=artifact_name, type="tensor")
        artifact.add_file(local_path=tensor_path)
        wandb.log_artifact(artifact)

    def log(self, metrics: dict):
        if self.no_logger:
            return
        wandb.log(metrics)

    def finish(self):
        if self.no_logger:
            return
        self.run.finish()
