Fooling around with an X-Ray image dataset

**Problem:** Multi-label chest X-ray classification - predict which of 15 thoracic diseases are present in an image (each image can have multiple labels).

**Dataset:** NIH Chest X-ray dataset (subset: 750 train / 250 test, full: 86k train / 25k test)
- 15 classes: No Finding, Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
- Massively imbalanced: "No Finding" is 58% of data, "Hernia" is 0.02%

<img width="1080" height="290" alt="download (11)" src="https://github.com/user-attachments/assets/c66bde07-a462-4c42-a55d-752e0bd2d0d7" />

**Model:** 
- DINOv3 (ViT-S/16) backbone + linear classification head, trained with BCE loss

<img width="1440" height="855" alt="1_Rm1yNe-lzd25Z1ETFvEmIg" src="https://github.com/user-attachments/assets/52057933-0390-4eab-bf2f-f3fb56cb9d78" />


**Attention scores visualisation:**

<img width="896" height="444" alt="1_0eFvS6mhhyG7o534azx3iQ" src="https://github.com/user-attachments/assets/115e4b0e-a0da-4ece-bb36-6c101979eb24" />

