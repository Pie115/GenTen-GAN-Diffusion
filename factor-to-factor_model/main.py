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

#import data loader and model
from data_loader import get_factors, extract_ecal
from model import FactorDiffusionModel, train_diffusion, sample_factors


def run_model(rank, num_samples, epochs = 200, device = 'cuda:1'):
    #extract data and run model, rank specifies the rank you would like factor matrices to be
    #num_samples is how many samples you want to train on, returns original data, and parameter count of model, as well as the model.

    tl.set_backend('pytorch')
    ecals_subset = extract_ecal(num_samples=num_samples, file_amount=1)
    ecal_train, ecal_test = train_test_split(ecals_subset, test_size=0.1, random_state = 1)
    #Keep random state the same so we can just store ecal_subset

    stored_factors, index_map, valid_indices = get_factors(ecals_subset=ecal_train, rank=rank)
    model = FactorDiffusionModel(rank=rank).to(device)
    parameter_count = train_diffusion(model, stored_factors, device=device, epochs=epochs)

    return model, ecals_subset, ecal_train, ecal_test, parameter_count

def generate_tensors(model, num_generated, rank, device = 'cuda:1'):
    #Generates new images, num_generated is how many tensors you would like to generate
    #returns the generated tensors
    A_gen, B_gen, C_gen = sample_factors(model, num_samples=num_generated, rank=rank, device=device)

    generated_tensors = [tl.cp_to_tensor((None, [A_gen[i], B_gen[i], C_gen[i]])) for i in range(num_generated)]

    return generated_tensors