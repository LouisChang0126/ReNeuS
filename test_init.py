import sys, os, torch, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from models.fields import SDFNetwork

sdf_net = SDFNetwork(d_in=3, d_out=257, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1, geometric_init=True, weight_norm=True, inside_outside=False)

# Test at origin (0, 0, 0)
origin = torch.tensor([[0.0, 0.0, 0.0]])
# Test at sphere boundary (0.5, 0, 0)
boundary = torch.tensor([[0.5, 0.0, 0.0]])
# Test outside (1.0, 0, 0)
outside = torch.tensor([[1.0, 0.0, 0.0]])

# Sample a line from origin to radius 1.5
x_vals = torch.linspace(0, 1.5, 20).unsqueeze(1)
pts = torch.zeros(20, 3)
pts[:, 0] = x_vals[:, 0]

with torch.no_grad():
    val_o = sdf_net.sdf(origin).item()
    val_b = sdf_net.sdf(boundary).item()
    val_out = sdf_net.sdf(outside).item()
    sdf_vals = sdf_net.sdf(pts).squeeze().numpy()

print(f"SDF at origin (r=0): {val_o:.4f}")
print(f"SDF at boundary (r=0.5): {val_b:.4f}")
print(f"SDF at outside (r=1.0): {val_out:.4f}")
print("SDF values along X-axis (0 to 1.5):")
print(np.round(sdf_vals, 3))
