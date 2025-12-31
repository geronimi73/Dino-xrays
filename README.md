Fooling around with an X-Ray image dataset

**Problem:** Multi-label chest X-ray classification - predict which of 15 thoracic diseases are present in an image (each image can have multiple labels).

**Dataset:** NIH Chest X-ray dataset (subset: 750 train / 250 test, full: 86k train / 25k test)
- 15 classes: No Finding, Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
- Massively imbalanced: "No Finding" is 58% of data, "Hernia" is 0.02%

**Model:** DINOv3 (ViT-S/16) backbone + linear classification head, trained with BCE loss

