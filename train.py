import torch
import torch.optim as optim
from model import CVAE 
from data_loader import load_data

def compute_loss(recon_data, data, mu, log_var):
    # Example loss function (modify as needed)
    #print('compute loss data shapes:')
    #print(recon_data.shape)
    #print(data.shape)
    beta = 10
    recon_loss = torch.nn.functional.binary_cross_entropy(recon_data, data, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_div = kl_div * beta
    ratio = kl_div/recon_loss
    return recon_loss + kl_div, ratio

def validate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_ratio = 0
    with torch.no_grad():  # No need to compute gradients during validation
        for data, condition in val_loader:
            data = data.to(device)
            condition = condition.to(device)
            recon_data, mu, log_var = model(data, condition)
            loss, val_ratio = compute_loss(recon_data, data, mu, log_var)
            total_loss += loss.item()
            total_ratio += val_ratio.item()
    avg_loss = total_loss / len(val_loader)
    avg_val_ratio = total_ratio / len(val_loader)
    return avg_loss, avg_val_ratio

def train(model, train_loader, optimizer, epochs=1000, device=torch.device("cuda")):
    model.to(device)
    model.train()
    total_ratio = 0
    for epoch in range(epochs):
        for i, (data, condition) in enumerate(train_loader):
            data = data.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            recon_data, mu, log_var = model(data, condition)
            loss, train_ratio = compute_loss(recon_data, data, mu, log_var)
            total_ratio += train_ratio.item()
            loss.backward()
            optimizer.step()
        val_loss, avg_val_ratio = validate(model, val_loader, device)
        avg_train_ratio = total_ratio/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {loss.item()},Training loss KL/MSE ratio: {avg_train_ratio}, Validation loss: {val_loss}, Validation loss KL/MSE ratio: {avg_val_ratio}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    train_loader, val_loader = load_data('normalized_file.csv')
    train(model, train_loader, optimizer, device=device)
