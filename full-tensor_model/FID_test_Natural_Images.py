import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from main import run_model, generate_tensors

import torch.nn.functional as F

def save_tensor_slices(tensor, save_path, file_name="generated_tensor.png", upscale_size=256):
    tensor = tensor.cpu().detach()
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, file_name)

    if tensor.shape[-1] == 25: 
        tensor = tensor.numpy()
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        for i in range(25):
            ax = axes[i // 5, i % 5]
            ax.imshow(tensor[:, :, i], cmap='viridis')
            ax.set_title(f"Slice {i}")
            ax.axis("off")
        plt.suptitle(file_name, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        plt.close()

    elif tensor.shape[-1] == 3: 
        img = tensor.permute(2, 0, 1).unsqueeze(0)
        img = (img - img.min()) / (img.max() - img.min())

        upscaled = F.interpolate(img, size=(upscale_size, upscale_size), mode='bilinear', align_corners=False)

        # Back to (H, W, 3)
        upscaled = upscaled.squeeze(0).permute(1, 2, 0).numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(upscaled)
        plt.axis("off")
        plt.title(file_name)
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        plt.close()

    else:
        raise ValueError(f"Unexpected tensor shape for saving: {tensor.shape}")

def run_FID(real_tensors, generated_tensors, dataset, device='cuda:1', batch_size=64):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    if dataset == "ecal":
        def normalize(t):
            t = (t - t.min()) / (t.max() - t.min())
            t = (t * 255).clamp(0, 255).to(torch.uint8)
            return t

        real = normalize(real_tensors).unsqueeze(1).repeat(1, 3, 1, 1, 1)
        fake = normalize(generated_tensors).unsqueeze(1).repeat(1, 3, 1, 1, 1)

        real = real.view(-1, 3, 51, 51)
        fake = fake.view(-1, 3, 51, 51)

        for i in range(0, len(real), batch_size):
            fid.update(real[i:i+batch_size].to(device), real=True)
        for i in range(0, len(fake), batch_size):
            fid.update(fake[i:i+batch_size].to(device), real=False)

    else:
        # Assume image dataset with 3-channel last dim
        def to_uint8(t):
            t = (t - t.min()) / (t.max() - t.min())
            return (t * 255).to(torch.uint8)

        real_rgb = to_uint8(real_tensors).permute(0, 3, 1, 2)
        fake_rgb = to_uint8(generated_tensors).permute(0, 3, 1, 2)

        for i in range(0, len(real_rgb), batch_size):
            fid.update(real_rgb[i:i+batch_size].to(device), real=True)
        for i in range(0, len(fake_rgb), batch_size):
            fid.update(fake_rgb[i:i+batch_size].to(device), real=False)

    score = fid.compute().item()
    print("FID:", score)
    return score

def fid_pipeline(dataset="ecal", num_samples=5000, epochs=200, n_repeats=5, device='cuda:1'):
    model, full_data, train_data, test_data, parameters = run_model(dataset=dataset, num_samples=num_samples, epochs=epochs, device=device)

    num_generated = int(0.10 * num_samples)
    num_comparisons = num_generated

    output_dir = f"generated_{dataset}_fid"
    os.makedirs(output_dir, exist_ok=True)

    fid_path = os.path.join(output_dir, "results.txt")
    fid_scores = []

    with open(fid_path, "w") as f:
        f.write(f"FID Scores, Parameters: {parameters}\n\n")

        for repeat in range(n_repeats):
            generated = generate_tensors(model, num_generated=num_generated, device=device)

            for i in range(min(10, len(generated))):
                save_tensor_slices(generated[i], output_dir, file_name=f"generated_tensor_{repeat}_{i}.png")
                print(f'Max Value: {generated[i].max()} Min Value: {generated[i].min()} Sum: {generated[i].sum()}')

            fid = run_FID(test_data[:num_comparisons], generated, dataset, device=device)
            fid_scores.append(fid)

            f.write(f"Repeat {repeat+1}: FID = {fid:.6f}\n")

            del generated
            gc.collect()
            torch.cuda.empty_cache()

        avg_fid = np.mean(fid_scores)
        std_fid = np.std(fid_scores)
        f.write(f"\nAverage FID: {avg_fid:.6f}\n")
        f.write(f"Standard Deviation: {std_fid:.6f}\n")

    print(f"[{dataset.upper()}] Params: {parameters} | Avg FID over {n_repeats} runs: {avg_fid:.4f}")

#fid_pipeline(dataset="celeba", epochs = 50, num_samples=10000)
#fid_pipeline(dataset="cifar10", epochs = 10, num_samples=4000)