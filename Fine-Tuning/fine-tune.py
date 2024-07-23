import torch
import torch.nn as nn
from tqdm import tqdm
import esm
import functions  # Assuming this module contains get_fasta_dict and SeqDataset
import torch.nn.functional as F
import os

# ESM alphabet
esm_alphabet = [
    '<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 
    'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 
    'Z', 'O', '.', '-', '<null_1>', '<mask>'
]

class ESMWithClassifier(nn.Module):
    def __init__(self, esm_model, alphabet_size):
        super().__init__()
        self.esm_model = esm_model
        self.classifier = nn.Linear(480, alphabet_size)  # Adjust to your alphabet size

    def forward(self, x):
        outputs = self.esm_model(x, repr_layers=[12])['representations'][12]
        return self.classifier(outputs.reshape(-1, 480))

def fine_tune_esm(fasta_file, alphabet_size, num_epochs, batch_size=64, learning_rate=0.0001):
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    fine_tune_model = ESMWithClassifier(esm_model, alphabet_size)
    
    # Load and prepare dataset using the loaded alphabet
    fasta_dict = functions.get_fasta_dict(fasta_file, 200, esm_alphabet)
    dataset = functions.SeqDataset(fasta_dict, 200)  # Ensure SeqDataset is compatible with ESM
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for param in fine_tune_model.parameters():
        param.requires_grad = False

    # Unfreeze the last two layers of the esm_model and classifier parameters
    for param in fine_tune_model.esm_model.layers[-1].parameters():
        param.requires_grad = True
    for param in fine_tune_model.esm_model.layers[-2].parameters():
        param.requires_grad = True
    for param in fine_tune_model.classifier.parameters():
        param.requires_grad = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fine_tune_model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fine_tune_model.parameters(), lr=learning_rate)
    mask_tok_idx = alphabet.get_idx('<mask>')  # Get the index of the mask token

    for epoch in range(num_epochs):
        fine_tune_model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            sequences = batch.to(device)
         
            # One-hot encode the sequences
            sequences_one_hot = F.one_hot(sequences, num_classes=len(alphabet))
           
            # Create a mask for the sequences
            mask = torch.rand(sequences.shape, device=device) < 0.15  # Randomly mask 15% of tokens
            labels = torch.full(sequences.shape, -100, device=device)  # Fill with -100
            labels[mask] = sequences[mask]
            mask = mask.unsqueeze(-1)
           
            # Apply the mask
            masked_input_one_hot = functions.apply_mask(sequences_one_hot, mask, mask_tok_idx)
            masked_input_indices = masked_input_one_hot.argmax(-1)
            
            optimizer.zero_grad()
            outputs = fine_tune_model(masked_input_indices)
           
            # Reshape for calculating Cross-Entropy Loss
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            print(f"Shape of outputs: {outputs.shape}")
            print(f"Shape of labels: {labels.shape}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

    # Save Model
    fasta_filename = os.path.basename(fasta_file)
    fasta_filename_without_ext = os.path.splitext(fasta_filename)[0]
    model_save_path = f'model_weights_freezed_{fasta_filename_without_ext}_32_e-4_12_40_Gua_corrected.pth'
    torch.save(fine_tune_model.state_dict(), model_save_path)

# Example usage
fine_tune_esm('Lysozyme.fasta', alphabet_size=len(esm_alphabet), num_epochs=40)
