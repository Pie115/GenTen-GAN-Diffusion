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

def run_FID(original_tensors, generated_tensors, num_comparisons, device='cuda:1'):
    generate_comparisons = torch.zeros((num_comparisons, 51, 51, 25), device=device)
    for i in range(num_comparisons):
        generate_comparisons[i] = generated_tensors[i] 

    print(generate_comparisons.shape) 

    generate_comparisons = generate_comparisons.permute(0, 3, 1, 2)
    print(generate_comparisons.shape)

    actual_comparisons = original_tensors[:num_comparisons].permute(0, 3, 1, 2)
    print(actual_comparisons.shape)  

    actual_norm = (actual_comparisons - actual_comparisons.min()) / (actual_comparisons.max() - actual_comparisons.min())
    gen_norm = (generate_comparisons - generate_comparisons.min()) / (generate_comparisons.max() - generate_comparisons.min())

    actual_uint8 = (actual_norm * 255).detach().cpu().numpy().astype(np.uint8)
    gen_uint8 = (gen_norm * 255).detach().cpu().numpy().astype(np.uint8)

    actual_t = torch.tensor(actual_uint8, dtype=torch.uint8)
    gen_t = torch.tensor(gen_uint8, dtype=torch.uint8)

    print("actual_comparisons_torch min/max:", actual_t.min().item(), actual_t.max().item())
    print("generated_tensors_torch min/max:", gen_t.min().item(), gen_t.max().item())
    print("actual_comparisons_torch shape:", actual_t.shape)
    print("generated_tensors_torch shape:", gen_t.shape)

    curr_sample = 0
    curr_slice = 0
    print("Sample value:", gen_t[curr_sample, curr_slice])

    fid = FrechetInceptionDistance(feature=2048).to(device)

    real_tensors = actual_t.unsqueeze(1).repeat(1, 3, 1, 1, 1).view(-1, 3, 51, 51).to(device)
    fake_tensors = gen_t.unsqueeze(1).repeat(1, 3, 1, 1, 1).view(-1, 3, 51, 51).to(device)

    batch_size = 64
    for i in range(0, real_tensors.shape[0], batch_size):
        fid.update(real_tensors[i:i+batch_size], real=True)
    for i in range(0, fake_tensors.shape[0], batch_size):
        fid.update(fake_tensors[i:i+batch_size], real=False)

    fid_score_tensors = fid.compute().item()
    print("Tensor-wise FID:", fid_score_tensors)
    return fid_score_tensors

n_repeats = 5
num_samples = 5000
num_generated = int(0.10 * num_samples)
num_comparisons = int(0.10 * num_samples)

model, original_tensors, ecal_train, ecal_test, parameters = run_model(num_samples=num_samples, epochs=50)
generated_tensors = generate_tensors(model=model, num_generated=num_generated)
save_directory = f"generated_tensors2"; os.makedirs(save_directory, exist_ok=True)

for i in range(min(10, len(generated_tensors))):
    print(f'Max Value: {generated_tensors[i].max()} Sum: {generated_tensors[i].sum()}')
    save_tensor_slices(generated_tensors[i], save_directory, file_name=f"generated_tensor_{i}.png")

fid_path = os.path.join(save_directory, "results.txt")
with open(fid_path, "w") as f:
    f.write(f"FID Scores, Parameters: {parameters}\n\n")
    fid_scores = []
    for repeat in range(n_repeats):
        fid_val = run_FID(original_tensors=ecal_test, generated_tensors=generated_tensors, num_comparisons=num_comparisons)
        fid_scores.append(fid_val)
        f.write(f"Repeat {repeat+1}: FID = {fid_val:.6f}\n")

        del generated_tensors
        torch.cuda.empty_cache()

        generated_tensors = generate_tensors(model=model, num_generated=num_generated)

    avg_fid = sum(fid_scores) / len(fid_scores)
    std_fid = np.std(fid_scores)
    f.write(f"\nAverage FID: {avg_fid:.6f}\n")
    f.write(f"Standard Deviation: {std_fid:.6f}\n")
print(f"[Parameters] {parameters} Completed {n_repeats} FID runs. Avg FID: {avg_fid:.4f}")