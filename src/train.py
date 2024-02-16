from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

from src.data.utils import prepare_data
from src.model import LanguageModel
from src.logger import WandbLogger
from src.utils import set_determinism
from configs.utils import update_sequence_mixer

import pyrallis
from src.config_pyr import TrainConfig as PyrTrainConfig


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
        gradient_accumulation_steps: int = 1,
        enable_tqdm: bool = False
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.tqdm = enable_tqdm

    def train_epoch(self, epoch_idx: int):
        self.model.train()

        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
            disable=not self.tqdm
        )
        self.optimizer.zero_grad()
        for idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # forward
            logits = self.model(inputs)
            
            # collect auxiliary losses
            auxiliary_loss = []

            def get_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    auxiliary_loss.append(module.get_auxiliary_loss())

            self.model.apply(get_auxiliary_loss)
            auxiliary_loss = sum(auxiliary_loss)

            # need to flatten batch and sequence dimensions
            main_loss = self.loss_fn(
                rearrange(logits, "... c -> (...) c"), targets.flatten()
            )
            loss = main_loss + auxiliary_loss
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            if (idx+1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # logging and printing
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log(
                {
                    "train/loss": loss,
                    "train/main_loss": main_loss,
                    "train/auxiliar_loss": auxiliary_loss,
                    "epoch": epoch_idx,
                }
            )
            global_step_idx = epoch_idx * len(self.train_dataloader) + idx
            self.logger.log_gamma_beta(self.model, global_step_idx)
            
        self.optimizer.zero_grad()

    def test(self, epoch_idx: int):
        self.model.eval()

        test_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)

                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), targets.flatten()
                )
                test_loss += loss / len(self.test_dataloader)

                # SE: important to
                all_preds.append(torch.argmax(logits, dim=-1).cpu())
                all_targets.append(targets.cpu())
                iterator.update(1)

            test_accuracy = compute_accuracy(
                torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)
            )

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }
            iterator.set_postfix(metrics)
            self.logger.log({"epoch": epoch_idx, **metrics})
        return metrics

    def fit(self):
        self.model.to("cuda")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )

        best_metric = 0.0
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)

            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > best_metric:
                self.logger.log_ckpt(self.model, tag="best")
                best_metric = metrics[self.early_stopping_metric]
                self.logger.run.summary["max_accuracy"] = best_metric

            # early stopping
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                self.logger.log_ckpt(self.model, tag=f"last_epoch_{epoch_idx}")
                break

            self.scheduler.step()


def compute_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
):
    return (preds == targets)[targets != ignore_index].to(float).mean()


@pyrallis.wrap()
def train(config: PyrTrainConfig):
    set_determinism(config.seed)
    config = update_sequence_mixer(config)

    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    train_dataloader, test_dataloader = prepare_data(config.data)
    model = LanguageModel(config=config.model)
    logger.log_model(model)
    
    print(model)
    print("#Parameters:", sum(param.numel() for param in model.parameters()))

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        enable_tqdm=config.tqdm
    )
    
    task.fit()
    logger.finish()


if __name__ == "__main__":
    train()
