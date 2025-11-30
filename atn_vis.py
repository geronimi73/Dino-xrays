import requests
import torch
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor
from transformers import AutoModel
from PIL import Image

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

def load_test_img():
  "Cats."
  url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
  img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
  
  return img

def plot_attention_overlay(image, attention_map, alpha=0.5, cmap='viridis'):
  "Overlay an image with attention heatmap"  
  plt.figure(figsize=(8, 6))
  plt.imshow(image)
  plt.imshow(attention_map, alpha=alpha, extent=[0, image.width, image.height, 0], cmap=cmap)
  plt.axis('off')
  fig = plt.gcf()
  plt.close()
  return fig
    
def run_test(
    device = "cuda",
    dino_repo = "facebook/dinov3-vits16-pretrain-lvd1689m",
    output_fn = "atn_vis_test.png"
  ):

  model = AutoModel.from_pretrained(dino_repo).to(device)
  # needed for extracting atn maps
  model.set_attn_implementation('eager')
  processor = AutoImageProcessor.from_pretrained(dino_repo)
  img = load_test_img()

  inputs = processor(img)["pixel_values"][0]
  with torch.no_grad():
    output = model(inputs.to(device).unsqueeze(dim=0), output_attentions=True)

  cls_attention = extract_cls_atn_map(output.attentions, model, processor)
  cls_attention_avg = cls_attention.mean(dim=0)
  
  plot = plot_attention_overlay(img, cls_attention_avg)
  plot.savefig(output_fn)

if __name__ == "__main__":
  run_test()
