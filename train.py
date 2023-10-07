import torch
import torch.optim as optim
from model import CVAE 
from data_loader import load_data
import pandas as pd
import os
import openpyxl

def compute_loss(recon_data, data, mu, log_var, burn_in_counter):
    # Example loss function (modify as needed)
    #print('compute loss data shapes:')
    #print(recon_data.shape)
    #print(data.shape)
    recon_loss = torch.nn.functional.binary_cross_entropy(recon_data, data, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_div = kl_div * burn_in_counter
    ratio = kl_div/recon_loss
    return recon_loss + kl_div, ratio, kl_div

def validate(model, val_loader, device, burn_in_counter):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_ratio = 0
    with torch.no_grad():  # No need to compute gradients during validation
        for data, condition in val_loader:
            data = data.to(device)
            condition = condition.to(device)
            recon_data, mu, log_var = model(data, condition)
            loss, val_ratio,kl_div = compute_loss(recon_data, data, mu, log_var, burn_in_counter)
            total_loss += loss.item()
            total_ratio += val_ratio.item()
    avg_loss = total_loss / len(val_loader)
    avg_val_ratio = total_ratio / len(val_loader)
    return avg_loss, avg_val_ratio, kl_div

def train(model, train_loader, optimizer, epochs=1000, device=torch.device("cuda")):
    model.to(device)
    model.train()
    
    # File name for the Excel file
    file_name = 'training_results.xlsx'
    
    # Check if the Excel file exists. If not, create one with headers.
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["Epoch", "Training Loss", "Training KL/MSE Ratio", "Validation Loss", "Validation KL/MSE Ratio", "train_kl_div", "val_kl_div"])
        with pd.ExcelWriter(file_name, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
    
    total_ratio = 0
    burn_in_counter = 0
    for epoch in range(epochs):
        for i, (data, condition) in enumerate(train_loader):
            data = data.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            recon_data, mu, log_var = model(data, condition)
            loss, train_ratio, train_kl_div = compute_loss(recon_data, data, mu, log_var, burn_in_counter)
            total_ratio += train_ratio.item()
            loss.backward()
            optimizer.step()
        val_loss, avg_val_ratio, val_kl_div = validate(model, val_loader, device, burn_in_counter)
        avg_train_ratio = total_ratio/len(train_loader)
        if epoch > 300 and burn_in_counter < 1.0:
            burn_in_counter += 0.003

        
        # Append the results of this epoch to the Excel file
        results_df = pd.DataFrame({
            "Epoch": [epoch+1],
            "Training Loss": [loss.item()],
            "Training KL/BCE Ratio": [avg_train_ratio],
            "Validation Loss": [val_loss],
            "Validation KL/BCE Ratio": [avg_val_ratio]
        })
        book = openpyxl.load_workbook(file_name)
        sheet = book.active
        train_kl_div_val = train_kl_div.cpu().item()
        val_kl_div_val = val_kl_div.cpu().item()
        # Append the results of this epoch to the worksheet
        row = [epoch+1, loss.item(), avg_train_ratio, val_loss, avg_val_ratio, train_kl_div_val, val_kl_div_val]
        sheet.append(row)
        # Save the workbook
        book.save(file_name)

        print(f"Epoch {epoch+1}/{epochs}, Training loss: {loss.item()},Training loss KL/BCE ratio: {avg_train_ratio}, Validation loss: {val_loss}, Validation loss KL/BCE ratio: {avg_val_ratio}, train_kl_div:{train_kl_div}, val_kl_div:{val_kl_div}")


if __name__ == "__main__":
    batch_size=32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    train_loader, val_loader = load_data('30knormalized_shuffled.csv', batch_size)
    train(model, train_loader, optimizer, device=device)
