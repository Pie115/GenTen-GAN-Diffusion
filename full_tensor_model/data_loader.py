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

# Filtering function: Removes events with total energy < 0.2 GeV as seen in original GAN paper (might not be necessary)
def filter_low_energy_events(tensor_batch, threshold=0.2):
    energy_sums = tensor_batch.sum(dim=(1, 2, 3))  # Sum over all cells
    mask = energy_sums >= threshold
    return tensor_batch[mask]

def reservoir_sampling(stream, k):
    reservoir = []
    for i, element in enumerate(stream):
        if (i < k):
            reservoir.append(element)
        else:
            j = random.randint(0,i)
            if (j < k):
                reservoir[j] = element
    return reservoir

def extract_ecal(file_amount = 1, device = 'cuda:1', num_samples = 1000):
    data_folder = "../datasets/calorimeter/gamma_random_angle/"

    # get one of the files from data_folder
    data_files = os.listdir(data_folder)

    ecal_data = []
    file_amount = file_amount #Change this to the amount of data files you want to grab, for everything change to len(data_files)
    print(f'Using {file_amount} files out of {len(data_files)}')

    for i in range(file_amount):
        file = f'{data_folder}{data_files[i]}'
        print(f"Trying file: {file}")

        try:
            # Try to open the file and extract ECAL data
            h5_file = h5py.File(file, 'r')
            file_keys = list(h5_file.keys())
            temp_tensor = np.array(h5_file['ECAL'])
            ecal_data.append(temp_tensor)
        except (OSError, KeyError) as e:
            print(f'Skipping file {file} due to error: {e}')
        
    ecal_data = np.array(ecal_data)
    ecal_data = ecal_data.reshape(-1, 51, 51, 25)  
    ecal_data.shape

    # Extract raw tensor data
    total_samples = num_samples + int(num_samples * 0.10) #Add test samples too assuming 90/10 train test split
    print(f'Randomly Sampled {num_samples} with {int(num_samples * 0.10)} testing data totalling {total_samples} samples')

    random.seed(1)    #Seed set so the same random sample is always selected

    ecal_sampled = reservoir_sampling(stream=ecal_data, k=total_samples)

    ecals_subset = torch.tensor(ecal_sampled, dtype=torch.float32, device=device)

    #Apply Filtering From GAN Paper
    #ecals_subset = filter_low_energy_events(ecals_subset)
    
    return ecals_subset