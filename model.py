from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
num_timesteps = 1000
betas = torch.linspace(1e-4, 0.02, num_timesteps)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

def forward_noise(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alpha_bar_t = (torch.sqrt(alphas_bar[t])).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = (torch.sqrt(1. - alphas_bar[t])).view(-1, 1, 1, 1)
    x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    return x_t, noise

def get_time_embedding(t, emb_dim=128):
    half_dim = emb_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

        self.time_proj = None
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb=None):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)

        if t_emb is not None and self.time_proj is not None:
            t_proj = self.time_proj(t_emb)[:, :, None, None]
            x = x + t_proj

        x = self.conv2(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_emb = nn.Linear(time_emb_dim, time_emb_dim)
        self.down1 = ConvBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.downsample = DownSample(base_channels * 2, base_channels * 4)

        self.upsample = UpSample(base_channels * 4, base_channels * 2)
        self.up1 = ConvBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up2 = ConvBlock(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
            t_emb = get_time_embedding(t)
            t_emb = self.time_emb(t_emb)

            d1 = self.down1(x, t_emb)
            d2 = self.down2(d1, t_emb)
            d3 = self.downsample(d2)

            u1 = self.upsample(d3)
            u1 = torch.cat([u1, d2], dim=1)
            u1 = self.up1(u1, t_emb)
            u1 = torch.cat([u1, d1], dim=1)
            u1 = self.up2(u1, t_emb)

            out = self.final_conv(u1)
            return out

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = UNET(in_channels=3, base_channels=64, time_emb_dim=128)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    global alphas_bar
    alphas_bar = alphas_bar.to(device)

    num_epochs = 300
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.shape[0]
            t = torch.randint(0, num_timesteps, (batch_size,), device=device)

            x_t, noise = forward_noise(images, t)
            pred_noise = model(x_t, t)

            loss = nn.MSELoss()(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.2f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"mod{epoch+1}.pth")
    print("Training complete!")

if __name__ == "__main__":
    train()