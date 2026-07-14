import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        labels = labels.long()

        ce_loss = F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        loss = (1.0 - pt).pow(self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                class_weights = torch.tensor(
                    [1.0 - self.alpha, self.alpha],
                    dtype=logits.dtype,
                    device=logits.device,
                )
            else:
                class_weights = torch.tensor(
                    self.alpha,
                    dtype=logits.dtype,
                    device=logits.device,
                )

            loss = loss * class_weights[labels]

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def get_loss_fn(
    loss_key: str,
    label_smoothing: float = 0.1,
    gamma: float = 2.0,
    alpha: float = None,
):
    loss_key = loss_key.lower().strip()

    if loss_key == "ce":
        return nn.CrossEntropyLoss()

    if loss_key == "label_smoothing":
        return nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

    if loss_key == "focal":
        return FocalLoss(
            gamma=gamma,
            alpha=alpha,
        )