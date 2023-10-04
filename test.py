import torch
from model import CVAE  # Ensure your model is in a file named cvae.py
from data_loader import load_test_data  # Replace with your test data loading function

def test(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            data, condition = batch  # Replace with your data and condition
            recon_data, mu, log_var = model(data, condition)
            loss = compute_loss(recon_data, data, mu, log_var)  # Define your loss function
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {avg_loss}")

if __name__ == "__main__":
    model = CVAE()
    model.load_state_dict(torch.load("cvae_model.pth"))  # Load the trained model
    test_loader = load_test_data()  # Replace with your test data loading function
    test(model, test_loader)
