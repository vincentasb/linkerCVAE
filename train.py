import torch
import torch.optim as optim
from model import CVAE 
from data_loader import load_data

def compute_loss(recon_data, data, mu, log_var):
    # Example loss function (modify as needed)
    #print('compute loss data shapes:')
    #print(recon_data.shape)
    #print(data.shape)
    recon_loss = torch.nn.functional.mse_loss(recon_data, data)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div

def validate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to compute gradients during validation
        for data, condition in val_loader:
            data = data.to(device)
            condition = condition.to(device)
            recon_data, mu, log_var = model(data, condition)
            loss = compute_loss(recon_data, data, mu, log_var)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train(model, train_loader, optimizer, epochs=1000, device=torch.device("cuda")):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for i, (data, condition) in enumerate(train_loader):
            data = data.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            recon_data, mu, log_var = model(data, condition)
            loss = compute_loss(recon_data, data, mu, log_var)
            loss.backward()
            optimizer.step()
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {loss.item()}, Validation loss: {val_loss}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    train_loader, val_loader = load_data('C:\\Users\\vincu\\Desktop\\codebase\\linkers\\pytorch\\CVAE\\normalized_file.csv')
    train(model, train_loader, optimizer, device=device)
