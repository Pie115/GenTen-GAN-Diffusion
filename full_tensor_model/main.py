import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
import torch
import tensorly as tl
from sklearn.model_selection import train_test_split

#import data loader and model
from data_loader import extract_ecal
from model import TensorDiffusionModel, train_tensor_diffusion, sample_tensors


def run_model(num_samples, epochs = 200, device = 'cuda:1'):
    #extract data and run model, rank specifies the rank you would like factor matrices to be
    #num_samples is how many samples you want to train on, returns original data, and parameter count of model, as well as the model.

    tl.set_backend('pytorch')
    ecals_subset = extract_ecal(num_samples=num_samples, file_amount=1)
    ecal_train, ecal_test = train_test_split(ecals_subset, test_size=0.1, random_state = 1)
    #Keep random state the same so we can just store ecal_subset

    model = TensorDiffusionModel().to(device)

    #No need to factorize on full model so we pass the samples directly
    parameter_count = train_tensor_diffusion(model, ecal_train, device=device, epochs=epochs)

    return model, ecals_subset, ecal_train, ecal_test, parameter_count

def generate_tensors(model, num_generated, device = 'cuda:1'):
    #Generates new images, num_generated is how many tensors you would like to generate
    #returns the generated tensors
    generated = sample_tensors(model, num_samples=num_generated, device=device)

    #Directly generating sample tensors
    return generated
