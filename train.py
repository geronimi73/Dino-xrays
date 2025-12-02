import torch
import os
import matplotlib.pyplot as plt
from collections import Counter
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, set_seed

from models import Dino3SingleDiseaseClassifier
from data import id2cls
from dataloader import create_weighted_dataloader
from utils import plot_losses

# cls2id = { "No Finding": 0, "Atelectasis": 1, "Cardiomegaly": 2, "Effusion": 3, "Infiltration": 4, "Mass": 5, "Nodule": 6, "Pneumonia": 7, "Pneumothorax": 8, "Consolidation": 9, "Edema": 10, "Emphysema": 11, "Fibrosis": 12, "Pleural_Thickening": 13, "Hernia": 14 }

def train(
  dino_repo = "facebook/dinov3-vits16-pretrain-lvd1689m",
  ds_repo = "g-ronimo/NIH-Chest-X-ray-dataset_resized300px",
  num_classes = 1,
  batch_size = 64,
  disease_id = 2,
  lr = 0.0001,
  epochs = 3,
  eval_steps = 700,       # eval ~2 per epoch
  device = "cuda",
  dtype = torch.bfloat16,
  freeze_backbone = False,
  augment_train_samples = True,
  seed = 7,
  output_dir = "models/",
  test = True,
  ):
  set_seed(seed)
  os.makedirs(output_dir, exist_ok=True)

  run_name = "disease-{disease_id}_epochs-{epochs}{augment}{frozen}".format(
    disease_id = disease_id,
    epochs = epochs,
    augment = '_augmented' if augment_train_samples else '',
    frozen = '_frozen' if freeze_backbone else '',
  )

  assert not os.path.exists(output_dir + run_name + ".txt"), f"Run already exists: {run_name}"

  log = partial(log_fn, log_dir = output_dir)
  log(run_name, f"seed: {seed}")
  log(run_name, f"Disease id: {disease_id} ({id2cls[disease_id]})")
  log(run_name, "Loading model and image processor ..")

  model = Dino3SingleDiseaseClassifier(dino_repo, num_classes, freeze_backbone=freeze_backbone).to(device).to(dtype)
  processor = AutoImageProcessor.from_pretrained(dino_repo)

  log(run_name, "Loading dataset ..")
  ds = load_dataset(ds_repo)
  if test:
    ds = ds["train"].shuffle(seed=seed).select(range(5_000)).train_test_split(seed=seed)

  log(run_name, "Creating dataloaders ..")
  dl_commons_args = dict(
    disease_id = disease_id,
    batch_size = batch_size,
    img_processor = processor,
    prefetch_factor = 8,
    num_workers = 8,
  )
  dl_train = create_weighted_dataloader(
    ds = ds["train"], 
    augment = augment_train_samples,
    **dl_commons_args
  )
  dl_test = create_weighted_dataloader(
    ds = ds["test"], 
    augment = False,
    **dl_commons_args
  )
  log(run_name, f"train: {len(dl_train)} batches, test: {len(dl_test)} batches (bs={batch_size})")

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

  metrics_train, metrics_eval = [], []
  step = 0
  max_acc = None
  for epoch in range(epochs):
    for inputs, labels in dl_train:
      inputs = inputs.to(model.device).to(model.dtype)
      labels = labels.to(model.device).to(model.dtype)
      
      loss, logits = model(inputs, labels)
      loss.backward()
      
      optimizer.step()
      optimizer.zero_grad()
    
      loss = loss.item()
      metrics_train.append(dict(step=step, loss=loss))
    
      if step % 10 == 0:
        log(run_name, f"epoch {epoch}, step {step}, loss {loss:.2f}")
      if step % (len(dl_train) - 1) == 0:
        model.eval()
        acc, loss_eval = eval(model, dl_test)
        metrics_eval.append(dict(step=step, acc=acc, loss=loss_eval))
        log(run_name, f"eval loss {loss_eval:.2f}, acc {acc:.2f}")
        model.train()

        if max_acc is None or acc > max_acc:
          max_acc = acc
          model.save_checkpoint(f"{output_dir}/{run_name}.pth")

      step += 1

  loss_plot = plot_losses(metrics_train, metrics_eval)
  loss_plot.savefig(f"{output_dir}/{run_name}_loss.png", dpi=150, bbox_inches='tight')     

def log_fn(run_name, text, log_dir):
  "Log to stdout and file logfile"
  print(text)
  with open(f"{log_dir}/{run_name}.txt", 'a', encoding='utf-8') as file:
    file.write(text + "\n")  

def eval(model, dl):
  "Evaluation step, returns accuracy and avg. loss on given dataset (dataloader `dl`)"
  cnt_all, cnt_correct, num_batches = 0, 0, 0
  loss_sum = 0
  preds_all = []
  for inputs, labels in tqdm(dl):
    num_batches += 1
    inputs, labels = inputs.to(model.device), labels.to(model.device)
    inputs, labels = inputs.to(model.dtype), labels.to(model.dtype)
    with torch.no_grad():
        loss, logits = model(inputs, labels)
    loss, logits, labels = loss.cpu(), logits.cpu(), labels.cpu()

    # count correct samples per batch for accuracy
    pred = (torch.sigmoid(logits) > 0.5).long()
    preds_all.extend(pred.squeeze().tolist())
    cnt_all += len(labels)
    cnt_correct += torch.sum(pred == labels.long()).item()

    # sum loss per batch
    loss_sum += loss.item()

  print(Counter(preds_all))  # Rough look at predictions

  return cnt_correct / cnt_all, loss_sum / num_batches

if __name__ == "__main__":
  train(
    disease_id = 2, 
    epochs = 2
  )
  # for disease_id in [d for d in range(0, 15)]:
    # train(disease_id = disease_id)
