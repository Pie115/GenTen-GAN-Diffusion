import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
import torch
import tensorly as tl
import pickle
import random
from tensorly.decomposition import parafac, non_negative_parafac
from torchvision import datasets, transforms


def filter_low_energy_events(tensor_batch, threshold=0.2):
    energy_sums = tensor_batch.sum(dim=(1, 2, 3))
    mask = energy_sums >= threshold
    return tensor_batch[mask]

def reservoir_sampling(stream, k):
    reservoir = []
    for i, element in enumerate(stream):
        if i < k:
            reservoir.append(element)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = element
    return reservoir

def extract_ecal(file_amount=1, device='cuda:1', num_samples=1000):
    data_folder = "../datasets/calorimeter/gamma_random_angle/"
    data_files = os.listdir(data_folder)
    ecal_data = []

    print(f'Using {file_amount} files out of {len(data_files)}')

    for i in range(file_amount):
        file = f'{data_folder}{data_files[i]}'
        print(f"Trying file: {file}")
        try:
            h5_file = h5py.File(file, 'r')
            temp_tensor = np.array(h5_file['ECAL'])
            ecal_data.append(temp_tensor)
        except (OSError, KeyError) as e:
            print(f'Skipping file {file} due to error: {e}')

    ecal_data = np.array(ecal_data).reshape(-1, 51, 51, 25)

    total_samples = num_samples + int(num_samples * 0.10)
    print(f'Randomly Sampled {num_samples} + {int(num_samples * 0.10)} test samples = {total_samples}')

    random.seed(1)
    ecal_sampled = reservoir_sampling(ecal_data, total_samples)
    ecals_subset = torch.tensor(ecal_sampled, dtype=torch.float32, device=device)

    return ecals_subset



def extract_celebA(data_dir="../datasets/celeba", device='cuda:1', num_samples=1000, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    dataset = datasets.CelebA(root=data_dir, split='train', download=True, transform=transform)

    selected_indices = torch.randperm(len(dataset))[:num_samples]
    images = torch.stack([dataset[i][0] for i in selected_indices]).to(device)

    images = images.permute(0, 2, 3, 1)

    print(f"Loaded {images.shape[0]} CelebA face images at {image_size} {image_size}")
    return images



def extract_cifar10_cats(data_dir="../datasets/cifar10", device='cuda:1', num_samples=1000):
    transform = transforms.ToTensor()
    
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    cat_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
    selected_indices = cat_indices[:num_samples]

    images = torch.stack([dataset[i][0] for i in selected_indices]).to(device)
    images = images.permute(0, 2, 3, 1) 

    print(f"Loaded {images.shape[0]} cat images from CIFAR-10")
    return images


def get_dataset(name="ecal", **kwargs):
    if name.lower() == "ecal":
        return extract_ecal(**kwargs)
    elif name.lower() == "cifar10":
        return extract_cifar10_cats(**kwargs)
    elif name.lower() == "celeba":
        return extract_celebA(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
