import torch
import torch.nn as nn
from transformers import AutoModel

class Dino3SingleDiseaseClassifier(nn.Module):
  def __init__(
    self, 
    backbone_repo, 
    num_classes = 1,
    freeze_backbone = True,
  ):
    super().__init__()
    self.backbone = AutoModel.from_pretrained(backbone_repo)
    self.backbone_repo = backbone_repo
    if freeze_backbone:
      for p in self.backbone.parameters():
        p.requires_grad = False
      self.backbone.eval()

    self.head = nn.Linear(
        self.backbone.config.hidden_size,
        num_classes
    )

  def forward(self, x, labels=None, return_dino_atn = False):
    x = self.backbone(x, output_attentions=return_dino_atn)
    atn = x.attentions[-1].cpu() if return_dino_atn else None 
    x = x.last_hidden_state[:,0]
    logits = self.head(x)

    if labels is not None:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits      
    return logits if not return_dino_atn else logits, atn
      
  @property
  def device(self): return self.backbone.device

  @property
  def dtype(self): return self.backbone.dtype

  def save_checkpoint(self, path, **kwargs):
    "Save model checkpoint with optional training state."
    checkpoint = {
      'model_state_dict': self.state_dict(),
      'model_config': {
          'num_classes': self.head.out_features,
          'backbone_repo': self.backbone_repo
      }
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

  @classmethod
  def load_checkpoint(cls, path, freeze_backbone=True):
    "Load model from checkpoint."
    checkpoint = torch.load(path)
    
    # Create model instance
    model = cls(
        backbone_repo=checkpoint['model_config']['backbone_repo'],
        num_classes=checkpoint['model_config']['num_classes'],
        freeze_backbone=freeze_backbone
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
        
    print(f"Checkpoint loaded from {path}")
    return model

