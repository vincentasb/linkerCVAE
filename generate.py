import torch
from model import CVAE
from embedding import OHAAE

def generate(model, condition, num_samples=32):
    model.eval()
    with torch.no_grad():

        stacked_condition = condition.repeat(num_samples, 1)
        # Sample from the latent space
        z = torch.randn(num_samples, 16)  # Assuming 16 is the size of your latent space
        
        # Concatenate the condition to the latent samples
        z_conditioned = torch.cat((z, stacked_condition), dim=1)
        
        # Pass through the decoder
        generated_samples = model.decoder(z_conditioned)
    return generated_samples

if __name__ == "__main__":
    model = CVAE()
    model.load_state_dict(torch.load("trained_model.pth"))  # Load the trained model
    condition = torch.tensor([0.9761143270470359, 0.5921704278497084, 0.5692884053843488, 0.21047004706062203, 0.2570003975742872, 0.05019928407511976, 0.20387956911395758, 0.11139856296487492, 0.3381496037358561, 0.2951564152179482, 0.5755004001485333, 0.4036521975094175, 0.714587356691418, 0.5376905868791385, 0.480178399135754, 0.12437267131018609, 0.1765908273870751, 0.1926416919720524, 0.1832969088447003, 0.16600954955411107, 0.12642919279706913, 0.1294086103523636, 0.1359832940781223, 0.12045453895557659, 0.03649903942674158, 0.1388764248122098, 0.2950254195588757, 0.20854704297471963, 0.3532892304298264, 0.40121201181802774])  # Replace with your condition
    ohaae = OHAAE()
    samples = generate(model, condition, num_samples=5)
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample}")
                # Determine the number of values in the tensor
        num_values = sample.numel()
        print(f"Number of values in the tensor: {num_values}")
        deembedded_sequence = ohaae.deembed_sequence(sample)
        print(f"Sample {i+1}: {deembedded_sequence}")