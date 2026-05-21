import torch
from model import UNET, get_time_embedding, alphas, betas, alphas_bar, num_timesteps

device = 'cpu'
model = UNET(in_channels=3, base_channels=64, time_emb_dim=128)
model.load_state_dict(torch.load("C:\\Users\\amrit\\Downloads\\mod50 (1).pth", map_location=device))
model.to(device)
model.eval()
@torch.no_grad()
def sample(model, num_samples=16, ddim_steps=50, device='cpu'):
    model.eval()
    x = torch.randn(num_samples, 3, 32, 32, device=device)
    step_range = torch.linspace(num_timesteps-1, 0, ddim_steps, dtype=torch.long, device=device)
    
    for i in range(len(step_range)-1):
        t = step_range[i]
        t_next = step_range[i+1]
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor)
        alpha_bar_t = alphas_bar[t]
        alpha_bar_t_next = alphas_bar[t_next]
        x = torch.sqrt(alpha_bar_t_next) * ((x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)) + torch.sqrt(1 - alpha_bar_t_next) * pred_noise
    x = (x + 1) / 2
    return torch.clamp(x, 0, 1)
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
samples = sample(model, num_samples=16, device='cpu')
grid = make_grid(samples, nrow=4)
plt.imshow(grid.permute(1,2,0).cpu().numpy(), cmap='gray')
plt.axis('off')
plt.show()