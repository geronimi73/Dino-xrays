import os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def list_files(directory, prefix = None, suffix = None):
  "List files in dir, filter with pre- and suffix"
  files = []

  for filename in os.listdir(directory):
    hasPrefix = True if prefix is None else filename.startswith(prefix)
    hasSuffix = True if suffix is None else filename.endswith(suffix)

    if hasPrefix and hasSuffix:
      files.append(os.path.join(directory, filename))

  return files

def label_pil(image, label, sz=224):
  "Show a single image+labels as plot. 224px is the default preprocessing resize for DinoV3"
  fig,ax = plt.subplots(figsize=(3,3))
  ax.imshow(image, cmap='gray')
  ax.set_title(label, fontsize=10, wrap=True)
  ax.axis('off')
  plt.tight_layout()
  plt.close(fig)

  return fig_to_pil(fig)
  # return fig

def fig_to_pil(fig):
  "Turn a matplotlib figure into a PIL"
  buf = BytesIO()
  fig.savefig(buf, format='png', bbox_inches='tight')
  buf.seek(0)
  return Image.open(buf)

def make_grid(images, rows=1, cols=None):
  "Make a grid of PIL images."
  if cols is None: cols = len(images)
  w, h = images[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  for i, image in enumerate(images):
      grid.paste(image, box=(i%cols*w, i//cols*h))
  return grid

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
