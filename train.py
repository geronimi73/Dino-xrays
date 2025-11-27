import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

def train(
  dino_repo = "facebook/dinov3-vits16-pretrain-lvd1689m",
  preprocessed_dataset = "NIH-Chest-X-ray_preprocessed",
  num_classes = 15,
  batch_size = 64,
  lr = 0.0001,
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

  print("Train!")
  model.train()

  step = 0
  for epoch in range(3):
    for inputs, labels in dl_train:
      inputs = inputs.to(model.device).to(model.dtype)
      labels = labels.to(model.device).to(model.dtype)
      
      output = model(inputs)
      loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
      loss.backward()
      
      optimizer.step()
      optimizer.zero_grad()
  
      loss = loss.item()
      if step % 10 == 0:
        print(f"epoch {epoch}, step {step}, loss {loss:.2f}")
      if step % 500 == 0:
        model.eval()
        accuracy(model, dl_eval)
        model.train()
      step += 1

def accuracy(model, dataloader):
  def logits_to_classes(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs > threshold).long()

  num_all, num_correct, loss_sum = 0, 0, 0
  for inputs, labels in tqdm(dataloader):
    with torch.no_grad():
      logits = model(inputs.to(model.device).to(model.dtype)).cpu()
      loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).cpu()

    labels_pred = logits_to_classes(logits)
    num_correct += (labels == labels_pred).all(dim=1).sum().item()
    num_all += inputs.shape[0]
    loss_sum += loss.item()
  print(f"eval loss {loss_sum/len(dataloader):.2f}, acc {num_correct/num_all:.2f}")

def get_dataloader(ds, bs):
  def collate_fn(items):
    images_tensor = torch.cat(
      [torch.Tensor(i["image_tensor"]) for i in items]
    )
    labels = [i["labels"] for i in items]
    labels_tensor = torch.zeros(len(labels), 15)
    for i,lbls in enumerate(labels): 
        labels_tensor[i, lbls] = 1
    
    return images_tensor, labels_tensor

  dl_common_args = dict(
    collate_fn = collate_fn,
    prefetch_factor = 2,
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
  # processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
  # ds = ds["train"].select(range(100)).train_test_split()

  def preprocess_single(item, resizeTo=(300, 300)):
    # not all images are RGB     
    # item["image_tensor"] = processor(item["image"].convert('RGB'))["pixel_values"]
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

if __name__ == "__main__":
  preprocess_xray_ds()
  # train()
