import torch
from model import CVAE  # Ensure your model is in a file named cvae.py

def generate(model, condition, num_samples=1):
    model.eval()
    with torch.no_grad():
        samples = []
        for _ in range(num_samples):
            z = torch.randn(model.latent_dim)  # Replace with your latent dimension
            sample = model.decoder(z, condition)
            samples.append(sample)
    return samples

if __name__ == "__main__":
    model = CVAE()
    model.load_state_dict(torch.load("cvae_model.pth"))  # Load the trained model
    condition = torch.tensor([1])  # Replace with your condition
    samples = generate(model, condition, num_samples=5)
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample}")
