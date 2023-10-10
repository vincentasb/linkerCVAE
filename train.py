import torch
import torch.optim as optim
from model import CVAE 
from data_loader import load_data
import pandas as pd
import os
import openpyxl

def compute_loss(recon_data, data, mu, log_var, burn_in_counter):
    #print('compute loss data shapes:')
    #print(recon_data.shape)
    #print(data.shape)
    recon_loss = torch.nn.functional.binary_cross_entropy(recon_data, data, reduction='sum') #BCE RECONSTRUCTION LOSS
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) #KL_DIV FOR SEEING HOW CLOSE LATENT SPACE IS TO A NORMAL DISTRIBUTION
    kl_div = kl_div * burn_in_counter
    ratio = kl_div/recon_loss #DEBUGGING RATIO, IGNORE
    return recon_loss + kl_div, ratio, kl_div, recon_loss

def validate(model, val_loader, device, burn_in_counter):
    model.eval()
    total_loss = 0 #DEBUGGING RATIO, IGNORE
    total_ratio = 0 #DEBUGGING RATIO, IGNORE
    with torch.no_grad():
        for data, condition in val_loader:
            data = data.to(device)
            condition = condition.to(device)
            recon_data, mu, log_var = model(data, condition)
            loss, val_ratio,kl_div, recon_loss = compute_loss(recon_data, data, mu, log_var, burn_in_counter)
            total_loss += loss.item() #DEBUGGING RATIO, IGNORE
            total_ratio += val_ratio.item() #DEBUGGING RATIO, IGNORE
    avg_loss = total_loss / len(val_loader)
    avg_val_ratio = total_ratio / len(val_loader) #DEBUGGING RATIO, IGNORE
    return avg_loss, avg_val_ratio, kl_div, recon_loss

def train(model, train_loader, optimizer, epochs=1000, device=torch.device("cuda")):
    model_save_path = "saved_model.pth"
    model.to(device)
    model.train()
    
    #INITIALIZING EXCELL FILE FOR LOSS LOGGING
    file_name = 'training_results.xlsx'
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["Epoch", "Training Loss", "Training KL/MSE Ratio", "Validation Loss", "Validation KL/MSE Ratio", "train_kl_div", "val_kl_div","train_recon", "val_recon"])
        with pd.ExcelWriter(file_name, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
    
    total_ratio = 0 #DEBUGGING RATIO, IGNORE
    burn_in_counter = 0.003
    try:
        for epoch in range(epochs):
            for i, (data, condition) in enumerate(train_loader):
                data = data.to(device)
                condition = condition.to(device)
                optimizer.zero_grad()
                recon_data, mu, log_var = model(data, condition)
                loss, train_ratio, train_kl_div, train_recon_loss = compute_loss(recon_data, data, mu, log_var, burn_in_counter)
                total_ratio += train_ratio.item() #DEBUGGING RATIO, IGNORE
                loss.backward()
                optimizer.step()
            val_loss, avg_val_ratio, val_kl_div, val_recon_loss = validate(model, val_loader, device, burn_in_counter)
            avg_train_ratio = total_ratio/len(train_loader) #DEBUGGING RATIO, IGNORE

            #BURN IN COUNTER FOR AVOIDING MODE COLLAPSE AND KL_DIV GOING TO 0 TOO FAST
            #if epoch >20 and burn_in_counter > 0.035:
            #    burn_in_counter -= 0.003

            
            # SCRIPT FOR APPENDING LOSSES TO AN EXCEL FILE FOR ANALYSIS
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
            train_recon_loss_val = train_recon_loss.cpu().item()
            val_recon_loss_val = val_recon_loss.cpu().item()
            row = [epoch+1, loss.item(), avg_train_ratio, val_loss, avg_val_ratio, train_kl_div_val, val_kl_div_val, val_recon_loss_val, train_recon_loss_val]
            sheet.append(row)
            book.save(file_name)
            # END OF SCRIPT FOR APPENDING LOSSES TO AN EXCEL FILE FOR ANALYSIS

            #CONSOLE LOSS LOGS
            print(f"Epoch {epoch+1}/{epochs}, Training loss: {loss.item()},Training loss KL/BCE ratio: {avg_train_ratio}, Validation loss: {val_loss}, Validation loss KL/BCE ratio: {avg_val_ratio}, train_kl_div:{train_kl_div}, val_kl_div:{val_kl_div}, train_recon:{train_recon_loss}, val_recon:{val_recon_loss_val}")

    except KeyboardInterrupt:
        print("Interrupted! Saving model state...")
        torch.save(model.state_dict(), model_save_path)
        return
    
    torch.save(model.state_dict(), model_save_path)
    print("Training completed! Model saved.")


if __name__ == "__main__":
    batch_size=32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    train_loader, val_loader = load_data('full_trainingdata.csv', batch_size)
    train(model, train_loader, optimizer, device=device)
