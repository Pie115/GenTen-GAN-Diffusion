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

#Performs CP decomposition on a batch of tensors and stores valid decompositions i.e. no error in factoring
def cp_decompose_batch(tensor_batch, rank, device = 'cuda:1'):
    tl.set_backend('pytorch')
    factors_list = []
    index_map = {}  # Maps original indices to stored positions
    stored_idx = 0

    with torch.no_grad():
        for i, tensor in enumerate(tensor_batch):
            try:
                #Perform CP decomposition (Nonnegative)
                tensor_factors = non_negative_parafac(tensor, rank=rank)
                #Extract factors (ignore weights)
                factors = [factor.to(device) for factor in tensor_factors.factors]

                factors_list.append(factors)
                index_map[i] = stored_idx  # Map valid sample index in case singular matrix error
                stored_idx += 1  

            except Exception as e:
                print(f"Skipping tensor {i} due to error: {e}")

    return factors_list, index_map


def clean_factors(stored_factors, index_map, device = 'cuda:1'):
    #Get rid of NAN values
    cleaned_factors = []
    updated_index_map = {}
    stored_idx = 0  # Re-indexing valid factors

    for i, factors in enumerate(stored_factors):
        if any(torch.isnan(f).any() for f in factors): 
            print(f"Removing NaN entry at index {i}")
            continue  # Skip this entry

        cleaned_factors.append(factors)
        updated_index_map[i] = stored_idx  # Map original index to new stored index
        stored_idx += 1

    stored_factors = cleaned_factors
    index_map = updated_index_map

    valid_indices = torch.tensor(list(index_map.keys()), device=device)
    print(f"Cleaned stored_factors. Remaining valid entries: {len(stored_factors)}")
    return stored_factors, index_map, valid_indices

def save_factors_to_cache(factors, index_map, valid_indices, rank, cache_dir="cached_factors_gammaRA_randomsample"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"rank_{rank}_factors.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "factors": factors,
            "index_map": index_map,
            "valid_indices": valid_indices.cpu().numpy()
        }, f)
    print(f"[Cache] Saved factors to {path}")

def load_factors_from_cache(rank, cache_dir="cached_factors_gammaRA_randomsample", device='cuda:1'):
    path = os.path.join(cache_dir, f"rank_{rank}_factors.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[Cache] Loaded cached factors for rank {rank} from {path}")
        # Move tensors to the correct device
        factors = [[f.to(device) for f in factor_group] for factor_group in data["factors"]]
        index_map = data["index_map"]
        valid_indices = torch.tensor(data["valid_indices"], device=device)
        return factors, index_map, valid_indices
    return None, None, None

def get_factors(ecals_subset, rank, device='cuda:1'):
    # Try loading cached factors
    factors, index_map, valid_indices = load_factors_from_cache(rank, device=device)
    if factors is not None:
        return factors, index_map, valid_indices

    # If not cached, compute them
    stored_factors, index_map = cp_decompose_batch(ecals_subset, rank, device=device)
    stored_factors, index_map, valid_indices = clean_factors(stored_factors, index_map, device=device)

    # Save to cache
    save_factors_to_cache(stored_factors, index_map, valid_indices, rank)

    return stored_factors, index_map, valid_indices