import requests
import torch
import matplotlib.pyplot as plt
import re
import os

from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor
from PIL import Image
from datasets import load_dataset

from data import id2cls
from models import Dino3SingleDiseaseClassifier
from dataloader import split_by_disease
from utils import list_files, make_grid, fig_to_pil, label_pil

def extract_cls_atn_map(attentions, model, processor):
  "List of pairwise attention scores -> NxM matrix"
  # height and width of atn map; depends on patch size and input img size
  atn_mat_h, atn_mat_w = [
    x / model.config.patch_size 
    for x in (processor.size["height"],processor.size["width"])
  ]
  assert all([x == int(x) for x in [atn_mat_h, atn_mat_w]])
  atn_mat_h, atn_mat_w

  # [B, head_num, tok1, tok2]
  cls_attention = attentions[0, :, 0, 5:].cpu()
  # -> [head_num, h, w]
  cls_attention = cls_attention.reshape(
    (model.config.num_attention_heads, int(atn_mat_h), int(atn_mat_w))
  )

  return cls_attention

def plot_attention_overlay(image, attention_map, title=None, alpha=0.5, cmap='viridis'):
  "Overlay an image with attention heatmap"  

  plt.figure(figsize=(8, 6))
  plt.imshow(image)
  plt.imshow(attention_map, alpha=alpha, extent=[0, image.width, image.height, 0], cmap=cmap)
  plt.axis('off')
  if title:
    plt.title(str(title), fontsize=14)
  fig = plt.gcf()
  plt.close()
  return fig
    
def process_image(img, model, processor):
  "Run an image through the model and return atn score avg. over all heads."
  inputs = processor(img)["pixel_values"][0]
  with torch.no_grad():
    logits, attentions = model(inputs.to(model.device).unsqueeze(dim=0), return_dino_atn=True)
  
  cls_attention = extract_cls_atn_map(attentions, model.backbone, processor)
  cls_attention_avg = cls_attention.mean(dim=0)

  return torch.sigmoid(logits).squeeze().item(), cls_attention_avg

def atn_vis(
  ckpt_path,
  num_samples = 20,
  ds_repo = "g-ronimo/NIH-Chest-X-ray-dataset_resized300px",
  ds_split = "test",
  device = "cuda",
):
  ""
  ds = load_dataset(ds_repo)[ds_split]
  disease_id = int(re.findall(f'disease-(\d+)', ckpt_path)[0])
  assert disease_id

  model = Dino3SingleDiseaseClassifier.load_checkpoint(ckpt_path)
  model = model.to(device)
  model.backbone.set_attn_implementation('eager')
  processor = AutoImageProcessor.from_pretrained(model.backbone_repo)

  print("Processing", ckpt_path)
  print(disease_id, id2cls[disease_id])

  pos_indcs, neg_indcs = split_by_disease(ds, disease_id)

  false_positives, false_negatives = [], []
  true_positives, true_negatives = [], []

  for i, idx in enumerate(tqdm(pos_indcs + neg_indcs)):
    # we done?
    if all([
      len(false_positives) > num_samples, len(false_negatives) > num_samples,
      len(true_positives) > num_samples, len(true_negatives) > num_samples,
    ]):
      break

    # skip pos. samples if we got all false_negatives + true_positives
    if all([
      len(false_negatives) > num_samples,
      len(true_positives) > num_samples,
    ]) and idx in pos_indcs:
      continue
      
    img, labels = ds[idx]["image"], ds[idx]["labels"]
    prob, atn = process_image(img, model, processor)
    
    prediction = prob > 0.5          # model's prediction: bool
    label = disease_id in labels     # actual label: bool
    
    fig = plot_attention_overlay(img, atn, title = f"{id2cls[disease_id]} ({disease_id}): {prob:.2f}")
    overview = make_grid([
      label_pil(img, f"#{idx} " + ", ".join([id2cls[id] for id in labels])).resize(img.size),
      fig_to_pil(fig).resize(img.size)
    ])

    if not label and prediction and not len(false_positives) > num_samples:
      false_positives.append(overview)
    elif label and not prediction and not len(false_negatives) > num_samples:
      false_negatives.append(overview)
    elif label and prediction and not len(true_positives) > num_samples:
      true_positives.append(overview)
    elif not label and not prediction and not len(true_negatives) > num_samples:
      true_negatives.append(overview)

  # Put images in results/run_name/*.png
  run_name_dir = os.path.dirname(ckpt_path)
  run_name = os.path.splitext(os.path.basename(ckpt_path))[0]
  output_dir = os.path.join(run_name_dir, run_name + "_dino-focus")
  os.makedirs(output_dir, exist_ok=True)

  for images, name in [
    (false_positives, "false_positives"),
    (false_negatives, "false_negatives"),
    (true_positives, "true_positives"),
    (true_negatives, "true_negatives"),
  ]:
    out_fn = f"{output_dir}/{name}.png"
    print(out_fn)
    make_grid(images, cols=3, rows=len(images)//3).save(out_fn)

if __name__ == "__main__":
  ckpt_dir = "./results"
  ckpt_files = list_files(ckpt_dir, suffix = ".pth")
  atn_vis(ckpt_files[0], num_samples=2)


