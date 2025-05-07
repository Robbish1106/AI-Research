import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# === 設定資料夾 ===
data_folder = os.path.abspath(os.path.join("..", "W3", "data"))
model_folder = os.path.abspath(os.path.join("..", "W3", "models"))
os.makedirs(model_folder, exist_ok=True)

# === VAE 模型定義 ===
class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 8, 8)
            nn.ReLU(),
            nn.Flatten(),  # (B, 128 * 8 * 8)
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (B, 1, 64, 64)
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        x_recon = self.decoder(self.decoder_input(z))
        return x_recon, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# === 資料集準備 ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = ImageFolder(data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 訓練設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=128).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 30
kld_weight = 0.01  # KLD Loss 權重

for epoch in range(epochs):
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        
        optimizer.zero_grad()
        recon_images, mu, logvar = vae(images)
        
        # 計算 Loss (重建誤差 + KLD)
        recon_loss = loss_fn(recon_images, images)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_weight * kld_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# === 儲存模型 ===
model_path = os.path.join(model_folder, "vae_model.pth")
torch.save(vae.state_dict(), model_path)
print(f"模型已儲存到 {model_path}")
