import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(
  dino_repo = "facebook/dinov3-vits16-pretrain-lvd1689m",
  preprocessed_dataset = "NIH-Chest-X-ray_preprocessed",
  num_classes = 15,
  batch_size = 128,
  lr = 0.0001,
  epochs = 10,
  device = "cuda",
  dtype = torch.bfloat16
):
  print("Loading model ..")
  model = create_dino_classifier(dino_repo, num_classes).to(device).to(dtype)
  print("Loading dataset ..")
  ds = load_preprocessed_ds(preprocessed_dataset)
  print("Creating dataloaders ..")
  dl_train, dl_eval = get_dataloader(ds, batch_size)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  print(f"train: {len(dl_train)} batches, test: {len(dl_eval)} batches (bs={batch_size})")

  print("Train!")
  model.train()

  pos_weight = [
    1.856301718650541,
    14.173754556500608,
    63.39673913043478,
    16.156509695290858,
    8.830431491294474,
    41.36524822695036,
    30.064432989690722,
    113.25242718446601,
    33.32857142857143,
    36.453125,
    112.16346153846153,
    54.004629629629626,
    45.213178294573645,
    40.36332179930796,
    555.4761904761905
  ]

  metrics_train, metrics_eval = [], []
  step = 0
  for epoch in range(epochs):
    for inputs, labels in dl_train:
      inputs = inputs.to(model.device).to(model.dtype)
      labels = labels.to(model.device).to(model.dtype)
      
      output = model(inputs)
      loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels, pos_weight= torch.sqrt(torch.Tensor(pos_weight).to(model.device)) )
      # loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
      loss.backward()
      
      optimizer.step()
      optimizer.zero_grad()
  
      loss = loss.item()
      metrics_train.append(dict(step=step, loss=loss))

      if step % 10 == 0:
        print(f"epoch {epoch}, step {step}, loss {loss:.2f}")
      if step % 500 == 0:
        model.eval()
        acc, loss_eval = accuracy(model, dl_eval)
        metrics_eval.append(dict(step=step, acc=acc, loss=loss_eval))
        print(f"eval loss {loss_eval:.2f}, acc {acc:.2f}")

        model.train()
      step += 1

  loss_plot = plot_losses(metrics_train, metrics_eval)
  loss_plot.savefig(f"loss_{epochs}-epochs_pos_weight_sqrt.png", dpi=150, bbox_inches='tight')     

def accuracy(model, dataloader):
  def logits_to_classes(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs > threshold).long()

  from collections import Counter
  import json

  examples = []
  num_all, num_correct, loss_sum = 0, 0, 0
  for inputs, labels in tqdm(dataloader):
    with torch.no_grad():
      logits = model(inputs.to(model.device).to(model.dtype)).cpu()
      loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).cpu()

    labels_pred = logits_to_classes(logits)
    for i in range(min(10, labels_pred.shape[0])):
      pred_indices = labels_pred[i].nonzero(as_tuple=True)[0].tolist()
      # true_indices = labels[i].nonzero(as_tuple=True)[0].tolist()
      examples.append(tuple(pred_indices))

    num_correct += (labels == labels_pred).all(dim=1).sum().item()
    num_all += inputs.shape[0]
    loss_sum += loss.item()

  print(json.dumps(Counter(examples).most_common(), indent=None))
  acc, loss = num_correct/num_all, loss_sum/len(dataloader)

  return acc, loss

def get_dataloader(ds, bs):
  processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

  def collate_fn(items):
    images_tensor = torch.stack(
      processor([ i["image"] for i in items ])["pixel_values"]
    )
    labels = [i["labels"] for i in items]
    labels_tensor = torch.zeros(len(labels), 15)
    for i,lbls in enumerate(labels): 
        labels_tensor[i, lbls] = 1
    
    return images_tensor, labels_tensor

  dl_common_args = dict(
    collate_fn = collate_fn,
    prefetch_factor = 8,
    num_workers = 8,
  )

  dl_train = DataLoader(ds["train"], batch_size = bs, **dl_common_args)
  dl_eval = DataLoader(ds["test"], batch_size = bs * 4, **dl_common_args)

  return dl_train, dl_eval

def load_preprocessed_ds(ds_name):
  return load_from_disk(ds_name)

def preprocess_xray_ds():
  ds = load_dataset(
    "alkzar90/NIH-Chest-X-ray-dataset", 
    'image-classification',
    num_proc = 12
  )

  def preprocess_single(item, resizeTo=(300, 300)):
    # not all images are RGB     
    item["image"] = item["image"].resize(resizeTo).convert('RGB')

    return item

  ds = ds.map(preprocess_single, batched=False, num_proc=12)
  # ds.set_format(type='torch', columns=['image_tensor', 'labels'], output_all_columns=True)
  ds.save_to_disk("NIH-Chest-X-ray_preprocessed")

def preprocess(item):
    item["image_tensor"] = processor(
        item["image"].convert('RGB') 
    )["pixel_values"]
    
    return item

def create_dino_classifier(dino_repo, num_classes):
  model = Dino3ChestXrayClassifier(
    backbone_repo = dino_repo,
    num_classes = num_classes,
  )

  return model

class Dino3ChestXrayClassifier(nn.Module):
  def __init__(
    self, 
    backbone_repo = "facebook/dinov3-vitb16-pretrain-lvd1689m", 
    num_classes = 15,
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

  def forward(self, x):
    x = self.backbone(x).last_hidden_state[:,0]
    x = self.head(x)
    
    return x 
      
  @property
  def device(self): return self.backbone.device

  @property
  def dtype(self): return self.backbone.dtype

# from model card; label -> id
cls2id = { "No Finding": 0, "Atelectasis": 1, "Cardiomegaly": 2, "Effusion": 3, "Infiltration": 4, "Mass": 5, "Nodule": 6, "Pneumonia": 7, "Pneumothorax": 8, "Consolidation": 9, "Edema": 10, "Emphysema": 11, "Fibrosis": 12, "Pleural_Thickening": 13, "Hernia": 14 }
# id -> labe;
id2cls = {}
for cls in cls2id: id2cls[cls2id[cls]] = cls

def show_item(item, sz=224):
  "Show a single image+labels as plot. 224px is the default preprocessing resize for DinoV3"
  thumb = item['image'].resize((sz,sz))
  # item['labels'] is a Tensor
  labels_txt = ', '.join([id2cls[lab] for lab in item['labels'].numpy()])
  fig,ax = plt.subplots(figsize=(3,3))
  ax.imshow(thumb, cmap='gray')
  ax.set_title(labels_txt, fontsize=10, wrap=True)
  ax.axis('off')
  plt.tight_layout()
  plt.close(fig)
  return fig

def plot_losses(losses_train, losses_eval):
  "Plot training and eval loss and accuracy."
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
  ax1.plot(
    [l["step"] for l in losses_train], 
    [l["loss"] for l in losses_train], 
    label='train loss',
    linewidth=1,
  )
  ax1.plot(
    [l["step"] for l in losses_eval], 
    [l["loss"] for l in losses_eval], 
    label='eval loss',
    linewidth=1,
  )
  ax1.set_xlabel('Step')
  ax1.set_ylabel('Loss')
  ax1.legend()
  ax1.set_title('Loss')
  
  ax2.plot(
    [l["step"] for l in losses_eval], 
    [l["acc"] for l in losses_eval], 
    linewidth=1,    
  )
  ax2.set_xlabel('Step')
  ax2.set_ylabel('Accuracy')
  ax2.set_title('Accuracy')
  
  plt.tight_layout()
  
  return fig

if __name__ == "__main__":
  # preprocess_xray_ds()
  train()
