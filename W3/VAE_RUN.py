import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from vae_test import VAE

# === 設定路徑 ===
data_folder = os.path.abspath(os.path.join("..", "W3", "data", "PLATEN_SH_IN_HDR_P_1"))
model_folder = os.path.abspath(os.path.join("..", "W3", "models"))
model_path = os.path.join(model_folder, "vae_model.pth")

os.makedirs(data_folder, exist_ok=True)

# === 設定 VAE 模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=128).to(device)
vae.load_state_dict(torch.load(model_path))
vae.eval()

# === 資料轉換 ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# === 處理單一圖片測試 ===
image_path = os.path.join(data_folder, "2021-01-01.png")  # 選擇一個測試圖片
img = Image.open(image_path).convert('L')
img_tensor = transform(img).unsqueeze(0).to(device)

# 預測
with torch.no_grad():
    recon_img, mu, logvar = vae(img_tensor)
    # 去掉多餘的維度
    recon_img = recon_img.squeeze().cpu().numpy()

# === 顯示結果 ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_tensor.cpu().squeeze(), cmap='gray')
plt.title(f"Original (2021-01-01)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(recon_img, cmap='gray')
plt.title(f"Reconstructed (2021-01-01)")
plt.axis('off')

plt.show()
