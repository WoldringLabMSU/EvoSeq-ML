import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(21, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64*404, 10000)
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, 2000)
        self.fc4 = nn.Linear(2000, 500)
        self.fc5 = nn.Linear(500, latent_dim)
        
        self.z_mean = nn.Linear(latent_dim, latent_dim)
        self.z_log_var = nn.Linear(latent_dim, latent_dim)
        self.sampling = Sampling()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

class Decoder(nn.Module):
    def __init__(self, latent_dim=200):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 500)
        self.fc2 = nn.Linear(500, 2000)
        self.fc3 = nn.Linear(2000, 5000)
        self.fc4 = nn.Linear(5000, 10000)
        self.fc5 = nn.Linear(10000, 64 * 404)
        
        self.deconv1 = nn.ConvTranspose1d(64, 21, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(21)
        self.dropout = nn.Dropout(0.2)  # Added dropout layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = x.view(x.size(0), 64, 404)
        x = torch.sigmoid(self.deconv1(x))
        return x.transpose(1, 2)

class ProteinVAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(ProteinVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction
    
    def loss(self, x, z_mean, z_log_var, reconstruction):
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return recon_loss + kl_loss

# Print the model
model = ProteinVAE()
print(model)