import sys, os, torch, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pyhocon import ConfigFactory
from exp_runner import Runner

conf_path = "confs/reneus.conf"
with open(conf_path) as f:
    conf_text = f.read().replace('./public_data/CASE_NAME/', '/home/louis/Fish-Dev/Dataset/3DGRUT/lego_glass/')

tmp_conf = '/tmp/reneus_test.conf'
with open(tmp_conf, 'w') as f:
    f.write(conf_text)

runner = Runner(tmp_conf, "train", "lego_glass", is_continue=True)

print("\n--- Model State ---")
print("SDF Network variance:", runner.deviation_network.variance.item())

rays_o, rays_d = runner.dataset.gen_rays_at(0, resolution_level=4)
rays_o = rays_o.reshape(-1, 3).cuda()
rays_d = rays_d.reshape(-1, 3).cuda()
near, far = runner.dataset.near_far_from_sphere(rays_o, rays_d)

with torch.no_grad():
    res = runner.renderer.render(rays_o, rays_d, near, far, extract_inner_render=True)

inner_color = res["inner_color_fine"]
print("\n--- Inner Render ---")
print("Inner color shape:", inner_color.shape)
print("Inner color min/max:", inner_color.min().item(), inner_color.max().item())
print("Inner color mean:", inner_color.mean().item())

weights = res["weights"]
print("\n--- Weights ---")
print("Weights sum min/max:", weights.sum(dim=-1).min().item(), weights.sum(dim=-1).max().item())

