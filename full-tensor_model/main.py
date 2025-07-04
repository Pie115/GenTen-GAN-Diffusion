import os
import torch
import tensorly as tl
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import h5py

# Import model and data loader
from data_loader import extract_ecal, extract_cifar10_cats, extract_celebA
from model import TensorDiffusionModel, train_tensor_diffusion, sample_tensors


def run_model(dataset='ecal', num_samples=5000, epochs=200, device='cuda:1'):
    tl.set_backend('pytorch')

    if dataset == 'ecal':
        tensors = extract_ecal(num_samples=num_samples, file_amount=2, device=device)
        tensor_shape = (51, 51, 25)
    elif dataset == 'cifar10':
        tensors = extract_cifar10_cats(num_samples=num_samples, device=device)
        tensor_shape = (32, 32, 3)
    elif dataset == 'celeba':
        tensors = extract_celebA(num_samples=num_samples, image_size=64, device=device)
        tensor_shape = (64, 64, 3)

    train_data, test_data = train_test_split(tensors, test_size=0.1, random_state=1)

    model = TensorDiffusionModel(tensor_shape=tensor_shape).to(device)
    param_count = train_tensor_diffusion(model, train_data, device=device, epochs=epochs)

    return model, tensors, train_data, test_data, param_count


def generate_tensors(model, num_generated, device='cuda:1'):
    generated = sample_tensors(model, num_samples=num_generated, device=device)
    return generated
