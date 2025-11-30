import re
import os 
from data import id2cls
from tabulate import tabulate
import matplotlib.pyplot as plt


num_samples_full_ds = {
  0: 50500,
  1: 8280,
  2: 1707,
  3: 8659,
  4: 13782,
  5: 4034,
  6: 4708,
  7: 876,
  8: 2637,
  9: 2852,
  10: 1378,
  11: 1423,
  12: 1251,
  13: 2242,
  14: 141
}

def list_txt_files(directory, prefix, suffix):
    files = []

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            files.append(os.path.join(directory, filename))

    return files

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def parse_log_files(prefix = "loss_", suffix = ".txt"):
    "Parse the train log files and extract accuracy+"
    files = list_txt_files(".", prefix = prefix, suffix = suffix)
    results = []

    for file in files:
        disease_id = re.findall(r'disease-(\d+)_', file)
        assert len(disease_id) == 1
        disease_id = int(disease_id[0])
        text = read_txt_file(file)
        max_acc = None
        for line in text.splitlines():
            acc_match = re.findall(f'eval.*acc\s(\d+\.\d+)', line)
            if acc_match:
                assert len(acc_match) == 1
                acc = float(acc_match[0])
                if max_acc is None or acc > max_acc:
                    max_acc = acc

        results.append(dict(
            disease_id=disease_id,
            disease = id2cls[disease_id],
            num_samples = num_samples_full_ds[disease_id],
            acc = max_acc,
            file = file
        ))

    # results.sort(key=lambda x:x["disease_id"])
    results.sort(key=lambda x:x["acc"], reverse=True)

    md_table = tabulate(
        results, 
        headers="keys",
        tablefmt="pipe"
    )
    print(md_table)
    plot_accuracies(results).savefig(f"log_analysis_{prefix}{suffix}.png", dpi=150, bbox_inches='tight')   

def plot_accuracies(data):
  "Barplot of top n occuring pathologies in dataset"
  accuracies = [x["acc"] for x in data]
  diseases = [f"{x['disease']} ({x['disease_id']})" for x in data]

  fig,ax = plt.subplots(figsize=(10,6))
  ax.barh(range(len(accuracies)), accuracies)
  ax.set_yticks(range(len(diseases)))
  ax.set_yticklabels(diseases, fontsize=9)
  ax.set_xlabel('Acc. [%]')
  ax.set_title(f'Max. Accuracy by disease')
  ax.invert_yaxis()
  for i,acc in enumerate(accuracies): 
    ax.text(acc, i, f' {acc*100:.0f}%', va='center', fontsize=8)
  plt.tight_layout()
  plt.close(fig)

  return fig



if __name__ == "__main__":
    parse_log_files(prefix = "loss_resized300px", suffix = "_augmented.txt")
