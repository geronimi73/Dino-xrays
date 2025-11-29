import torch.nn as nn
from transformers import AutoModel

class Dino3SingleDiseaseClassifier(nn.Module):
  def __init__(
    self, 
    backbone_repo = "facebook/dinov3-vitb16-pretrain-lvd1689m", 
    num_classes = 1,
    freeze_backbone = True,
  ):
    super().__init__()
    self.backbone = AutoModel.from_pretrained(backbone_repo)
    if freeze_backbone:
      for p in self.backbone.parameters():
        p.requires_grad = False
      self.backbone.eval()

    self.head = nn.Linear(
        self.backbone.config.hidden_size,
        num_classes
    )

  def forward(self, x, labels=None):
    x = self.backbone(x).last_hidden_state[:,0]
    logits = self.head(x)

    if labels is not None:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits      
    return logits
      
  @property
  def device(self): return self.backbone.device

  @property
  def dtype(self): return self.backbone.dtype
