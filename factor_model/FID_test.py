import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
import torch
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac
from torchmetrics.image.fid import FrechetInceptionDistance
from main import run_model, generate_tensors

def save_tensor_slices(tensor, save_path, file_name="generated_tensor.png"):
    #Save generated samples to file
    tensor = tensor.cpu().detach().numpy()  
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    for i in range(25):
        ax = axes[i // 5, i % 5]
        ax.imshow(tensor[:, :, i], cmap='viridis')
        ax.set_title(f"Slice {i}")
        ax.axis("off")

    plt.suptitle(file_name, fontsize=16)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True) 
    save_file = os.path.join(save_path, file_name)
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"Saved {save_file}")

def run_FID(original_tensors, generated_tensors, num_comparisons, device = 'cuda:1'):
    # Create an empty tensor to store the generated data
    generate_comparisons = torch.zeros((num_comparisons, 51, 51, 25), device=device)

    for i in range(num_comparisons):
        generate_comparisons[i] = generated_tensors[i] 

    print(generate_comparisons.shape) 

    generate_comparisons = generate_comparisons.permute(0, 3, 1, 2)  # Moves 25 to the second position
    print(generate_comparisons.shape)

    actual_comparisons = original_tensors[:num_comparisons]

    actual_comparisons = actual_comparisons.permute(0, 3, 1, 2)  

    print(actual_comparisons.shape)  

    actual_comparisons_norm = (actual_comparisons - actual_comparisons.min()) / (actual_comparisons.max() - actual_comparisons.min())
    generated_tensors_norm = (generate_comparisons - generate_comparisons.min()) / (generate_comparisons.max() - generate_comparisons.min())

    # Detach tensors, move to CPU, and convert to numpy
    actual_comparisons_uint8 = (actual_comparisons_norm * 255).detach().cpu().numpy().astype(np.uint8)
    generated_tensors_uint8 = (generated_tensors_norm * 255).detach().cpu().numpy().astype(np.uint8)

    # Convert back to PyTorch uint8 tensors
    actual_comparisons_torch = torch.tensor(actual_comparisons_uint8, dtype=torch.uint8)
    generated_tensors_torch = torch.tensor(generated_tensors_uint8, dtype=torch.uint8)

    # Print stats
    print("actual_comparisons_torch min/max:", actual_comparisons_torch.min().item(), actual_comparisons_torch.max().item())
    print("generated_tensors_torch min/max:", generated_tensors_torch.min().item(), generated_tensors_torch.max().item())
    print("actual_comparisons_torch shape:", actual_comparisons_torch.shape)
    print("generated_tensors_torch shape:", generated_tensors_torch.shape)

    # Sample check
    curr_sample = 0  # Example index
    curr_slice = 0   # Example index
    print("Sample value:", generated_tensors_torch[curr_sample, curr_slice])

    fid = FrechetInceptionDistance(feature=2048)

    real_tensors = actual_comparisons_torch.unsqueeze(1).repeat(1, 3, 1, 1, 1) 
    fake_tensors = generated_tensors_torch.unsqueeze(1).repeat(1, 3, 1, 1, 1) 

    real_tensors = real_tensors.view(-1, 3, 51, 51) 
    fake_tensors = fake_tensors.view(-1, 3, 51, 51) 
    fid.update(real_tensors, real=True)
    fid.update(fake_tensors, real=False)

    fid_score_tensors = fid.compute().item()
    print("Tensor-wise FID:", fid_score_tensors)
    return fid_score_tensors


ranks = [200, 250]
n_repeats = 5  # Number of times to repeat FID per rank
num_samples = 5000 #number of samples to train on

num_generated = int(0.10 * num_samples) #number of tensors to generate
num_comparisons = int(0.10 * num_samples) #number of samples to compare (should be = num generate)

for rank in ranks:
    model, original_tensors, ecal_train, ecal_test, parameters = run_model(rank=rank, num_samples=num_samples, epochs=200)

    generated_tensors = generate_tensors(model=model, num_generated=num_generated, rank=rank)

    save_directory = f"generated_tensors2_{rank}"
    os.makedirs(save_directory, exist_ok=True)

    # Save first 10 tensors
    for i in range(min(10, len(generated_tensors))):
        print(f'Max Value: {generated_tensors[i].max()} Sum: {generated_tensors[i].sum()}')
        save_tensor_slices(generated_tensors[i], save_directory, file_name=f"generated_tensor_{i}.png")

    fid_path = os.path.join(save_directory, "results.txt")
    with open(fid_path, "w") as f:
        f.write(f"FID Scores for Rank: {rank}, Parameters: {parameters}\n\n")
        fid_scores = []

        for repeat in range(n_repeats):
            #num_comparisons <= num_generated <= num_samples
            fid = run_FID(original_tensors=ecal_test, generated_tensors=generated_tensors, num_comparisons=num_comparisons)
            fid_scores.append(fid)
            f.write(f"Repeat {repeat+1}: FID = {fid:.6f}\n")

            generated_tensors = generate_tensors(model=model, num_generated=num_generated, rank=rank)

        avg_fid = sum(fid_scores) / len(fid_scores)
        f.write(f"\nAverage FID: {avg_fid:.6f}\n")

        std_fid = np.std(fid_scores)
        f.write(f"Standard Deviation: {std_fid:.6f}\n")


    print(f"[Rank {rank}] [Parameters] {parameters} Completed {n_repeats} FID runs. Avg FID: {avg_fid:.4f}")