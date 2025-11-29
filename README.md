
**Problem:** Multi-label chest X-ray classification - predict which of 15 thoracic diseases are present in an image (each image can have multiple labels).

**Dataset:** NIH Chest X-ray dataset (subset: 750 train / 250 test, full: 86k train / 25k test)
- 15 classes: No Finding, Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
- Massively imbalanced: "No Finding" is 58% of data, "Hernia" is 0.02%

**Model:** DINOv3 (ViT-S/16) backbone + linear classification head, trained with BCE loss

**Approaches tried:**

1. **Baseline (frozen backbone):** Model quickly plateaued at 33% exact-match accuracy, predicting almost exclusively "No Finding" or nothing

2. **pos_weight (inverse frequency):** Training became unstable with extreme weights (1.8 to 555), model learned nothing

3. **pos_weight (sqrt scaled):** Training stable, but model still predicted mostly "No Finding"

4. **Balanced sampling (WeightedRandomSampler):** Filtered to 16 combinations with ≥500 samples, sampled uniformly across combinations. Frozen backbone: model predicted empty []

5. **Balanced sampling + unfrozen backbone:** **Best result so far!** Model predicts diverse classes, achieves ~4-5% exact-match accuracy on full test set. Still predicts empty for many samples, gap between train/eval loss suggests overfitting

**Current status:** Model is learning but needs further tuning (threshold adjustment, per-label metrics, data augmentation, learning rate schedule)
