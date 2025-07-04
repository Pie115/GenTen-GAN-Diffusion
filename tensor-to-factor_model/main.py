import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
import torch
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac
from sklearn.model_selection import train_test_split

from data_loader import get_factors, extract_ecal
from model import TensorDiffusionModel, train_tensor_diffusion, sample_tensors


def run_model(rank, num_samples, epochs = 200, device = 'cuda:1'):
    #extract data and run model, rank specifies the rank you would like factor matrices to be
    #num_samples is how many samples you want to train on, returns original data, and parameter count of model, as well as the model.

    tl.set_backend('pytorch')
    tensors = extract_ecal(num_samples=num_samples, file_amount=5, device=device)
    tensor_shape = (51, 51, 25)
    train_data, test_data = train_test_split(tensors, test_size=0.1, random_state=1)

    model = TensorDiffusionModel(tensor_shape=tensor_shape, rank=rank).to(device)
    param_count = train_tensor_diffusion(model, train_data, device=device, epochs=epochs)
    #torch.save(model.state_dict(), f"tensor_diffusion_rank{rank}.pth")

    return model, tensors, train_data, test_data, param_count


def generate_tensors(model, num_generated, rank, device = 'cuda:1'):
    #Generates new images, num_generated is how many tensors you would like to generate
    #returns the generated tensors
    generated_tensors = sample_tensors(model, num_samples=num_generated, device=device)

    return generated_tensors
