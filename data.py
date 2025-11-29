from collections import Counter
from tqdm import tqdm

import matplotlib.pyplot as plt

# from model card; label -> id
cls2id = { "No Finding": 0, "Atelectasis": 1, "Cardiomegaly": 2, "Effusion": 3, "Infiltration": 4, "Mass": 5, "Nodule": 6, "Pneumonia": 7, "Pneumothorax": 8, "Consolidation": 9, "Edema": 10, "Emphysema": 11, "Fibrosis": 12, "Pleural_Thickening": 13, "Hernia": 14 }
# id -> labe;
id2cls = {}
for cls in cls2id: id2cls[cls2id[cls]] = cls

def show_item(item, sz=224):
  "Show a single image+labels as plot. 224px is the default preprocessing resize for DinoV3"
  thumb = item['image'].resize((sz,sz))
  labels_txt = ', '.join([id2cls[lab] for lab in item['labels']])
  fig,ax = plt.subplots(figsize=(3,3))
  ax.imshow(thumb, cmap='gray')
  ax.set_title(labels_txt, fontsize=10, wrap=True)
  ax.axis('off')
  plt.tight_layout()
  plt.close(fig)
  return fig

def plot_topn_pathologies(ds, top_n=20, splits=["train", "test"]):
  "Barplot of top n occuring pathologies in dataset"
  cnt = Counter()

  for split in splits:
    if split in ds:
      for row in tqdm(ds[split], f"{split}"): 
        cnt[tuple(row["labels"])] += 1
    else:
      print(f"Split {split} does not exist in dataset")

  sorted_combos = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
  labels_list = [', '.join([id2cls[l] for l in combo]) for combo,count in sorted_combos[:top_n]]
  counts_list = [count for combo,count in sorted_combos[:top_n]]
  fig,ax = plt.subplots(figsize=(10,6))
  ax.barh(range(len(counts_list)), counts_list)
  ax.set_yticks(range(len(labels_list)))
  ax.set_yticklabels(labels_list, fontsize=9)
  ax.set_xlabel('Count')
  ax.set_title(f'Top {top_n} Most Common Pathologies (Combinations)')
  ax.invert_yaxis()
  for i,count in enumerate(counts_list): 
    ax.text(count, i, f' {count}', va='center', fontsize=8)
  plt.tight_layout()
  plt.close(fig)

  return fig

