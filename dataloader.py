import numpy as np
import itertools 
import torch
import torchvision.transforms as T

from data import id2cls
from functools import partial
from torch.utils.data import WeightedRandomSampler, DataLoader

def split_by_disease(ds, disease_id, num_proc=2):
    "Split an NIH dataset by disease. Returns idcs of positive and negative items for given disease."
    def has_disease(item, disease_id):
        return dict(has_disease = disease_id in item["labels"])
    ds_split = ds.map(
        partial(has_disease, disease_id=disease_id), 
        num_proc=num_proc, 
        remove_columns=ds.features
    )
    pos_indcs = [idx for idx, item in enumerate(ds_split) if item["has_disease"]]
    neg_indcs = [idx for idx, item in enumerate(ds_split) if not item["has_disease"]]
    return pos_indcs, neg_indcs

def create_weighted_sampler(ds, disease_id, num_proc=2):
    "Create a weighted sampler for a given disease id, will return samples 50% with and 50% without disease"
    pos_indcs, neg_indcs = split_by_disease(ds, disease_id, num_proc=num_proc)
    weights = np.zeros(len(ds))
    weights[pos_indcs] = 1/len(pos_indcs)
    weights[neg_indcs] = 1/len(neg_indcs)

    print(f"create_weighted_sampler() for disease {disease_id}: {len(pos_indcs)}+ve samples ({len(pos_indcs)/len(ds)*100:.2f}%), {len(neg_indcs)}-ve samples")

    sampler = WeightedRandomSampler(weights, num_samples = len(ds))
    
    return sampler

def test_weighted_sampler(ds, disease_id, num_proc=2):
    "Test if create_weighted_sampler() is balanced"
    pos_indcs, neg_indcs = split_by_disease(ds, disease_id, num_proc=num_proc)
    sampler = create_weighted_sampler(ds, disease_id)

    sample_indices = list(itertools.islice(sampler, 1000))
    pos_count = sum(1 for idx in sample_indices if idx in pos_indcs)
    print(f"disease_id {disease_id} ({id2cls[disease_id]})")
    print(f"Positive: {pos_count}/1000 = {pos_count/10}%")
    assert pos_count > 450 and pos_count <550

def augment_img(img, resizeTo=(280, 280)):
    "Augment an xray image by rotating, transforming, changing brightness and color"
    transform = T.Compose([
        T.RandomRotation([-5,5]),
        T.CenterCrop(resizeTo),
        T.RandomResizedCrop(resizeTo, scale=(0.9, 1.0)),
        T.ColorJitter(brightness=0.1, contrast=0.2)
    ])
    return transform(img)

def collate_fn(items, disease_id, img_processor, augment):
    "Turn a list of NIH samples into a batch."
    images = [ i["image"] for i in items ]
    if augment:
        images = [ augment_img(img) for img in images ]        

    images_tensor = torch.stack(
      img_processor(images)["pixel_values"]
    )
    labels_tensor = torch.Tensor([
        (1 if disease_id in i["labels"] else 0) 
        for i in items
    # labels are currently shape [batch_size] as floats. For BCEWithLogitsLoss, you typically want them as [batch_size, 1] to match the model output shape. You can fix this by adding .unsqueeze(1) to the labels tensor:
    ]).unsqueeze(1) 

    return images_tensor, labels_tensor

def create_weighted_dataloader(
    ds, 
    batch_size, 
    disease_id, 
    img_processor,
    augment,
    prefetch_factor = None, 
    num_workers = 0,
):
    "Creates a balanced dataloader, returns batches where samples 50% w/ and 50% w/o disease"
    sampler = create_weighted_sampler(ds, disease_id)
    
    return DataLoader(
        ds, 
        collate_fn = partial(collate_fn, img_processor = img_processor, disease_id = disease_id, augment = augment),
        batch_size=batch_size, 
        sampler=sampler, 
        prefetch_factor=prefetch_factor, 
        num_workers=num_workers
    )
