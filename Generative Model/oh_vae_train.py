import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from vae_oh_CNN import ProteinVAE
import os
import random
from Bio import SeqIO

def one_hot_encode(sequence, amino_acids="ACDEFGHIKLMNPQRSTVWY-"):
    encoding = torch.zeros(len(sequence), len(amino_acids))
    for i, aa in enumerate(sequence):
        position = amino_acids.index(aa)
        encoding[i, position] = 1.0
    return encoding

def read_fasta_sequences(fasta_file):
    """
    Read sequences from a FASTA file and shuffle them.
    """
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    sequences = pd.Series(sequences).sample(frac=1).tolist()
    return sequences

def train_val_split(data, val_ratio=0.1):
    """
    Split data into training and validation sets.
    """
    shuffled_data = random.sample(data, len(data))
    val_size = int(len(shuffled_data) * val_ratio)
    val_data = shuffled_data[:val_size]
    train_data = shuffled_data[val_size:]
    return train_data, val_data

def train_vae_model(vae, train_loader, val_loader, device, num_epochs=300, lr=0.0001, weight_decay=1e-8, patience=5):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    lowest_val_loss = float('inf')
    patience_counter = 0
    prev_best_model_filename = None

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            z_mean, z_log_var, reconstruction = vae(batch)
            loss = vae.loss(batch, z_mean, z_log_var, reconstruction)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z_mean, z_log_var, reconstruction = vae(batch)
                val_loss += vae.loss(batch, z_mean, z_log_var, reconstruction).item()
        
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        
        if val_loss < lowest_val_loss:
            patience_counter = 0
            lowest_val_loss = val_loss
            if prev_best_model_filename and os.path.exists(prev_best_model_filename):
                os.remove(prev_best_model_filename)
            
            filename = f"Bali_vae_oh_latent100_16_1CNN_{lowest_val_loss:.6f}.pth"
            torch.save(vae.state_dict(), filename)
            print(f"Epoch {epoch+1}/{num_epochs}: New best model saved with Validation Loss: {lowest_val_loss:.6f}")
            prev_best_model_filename = filename
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Epoch {epoch+1}/{num_epochs}: Early stopping due to no improvement in validation loss.")
                break

def main():
    fasta_file = 'cleaned-aligned-sampled.fasta'
    sequences = read_fasta_sequences(fasta_file)
    print(f"Sequence length: {len(sequences[1])}")
    
    encoded_data = [one_hot_encode(seq) for seq in sequences]
    X_train, X_val = train_val_split(encoded_data, val_ratio=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = ProteinVAE().to(device)

    train_loader = DataLoader(X_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=16, shuffle=False)

    train_vae_model(vae, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
