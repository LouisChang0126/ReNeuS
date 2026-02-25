import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic


# ==============================================================================
# [ReNeuS] 新增：光線追蹤工具函數 (論文 Sec 3.3)
# NeuS 原版不包含以下函數。ReNeuS 新增了：
#   - compute_refraction(): Snell's Law 折射計算 (Eq.7)
#   - compute_reflection(): 反射定律 (Eq.7)
#   - compute_fresnel(): Fresnel 方程式, natural light assumption (Eq.8-9)
#   - check_total_internal_reflection(): TIR 檢查
#   - ray_mesh_intersection(): trimesh 的光線-容器交叉檢測
# ==============================================================================

def compute_refraction(ray_d, normal, ior_in, ior_out):
    """
    Compute refraction direction using Snell's Law.
    
    Args:
        ray_d: [N, 3] incident ray direction (normalized)
        normal: [N, 3] surface normal (normalized, pointing outward)
        ior_in: [N] or scalar, IOR of incident medium
        ior_out: [N] or scalar, IOR of refracted medium
    
    Returns:
        ray_d_refract: [N, 3] refracted ray direction (normalized)
        valid_mask: [N] bool tensor, False if total internal reflection occurs
    """
    # Ensure inputs are normalized
    ray_d = F.normalize(ray_d, dim=-1)
    normal = F.normalize(normal, dim=-1)
    
    # Compute cos(theta_i)
    cos_theta_i = -(ray_d * normal).sum(dim=-1, keepdim=True)
    
    # Handle rays hitting from inside (flip normal if needed)
    # If cos_theta_i < 0, ray is hitting from inside, flip normal
    normal = torch.where(cos_theta_i < 0, -normal, normal)
    cos_theta_i = cos_theta_i.abs()
    
    # IOR ratio
    if not isinstance(ior_in, torch.Tensor):
        ior_in = torch.tensor(ior_in, device=ray_d.device)
    if not isinstance(ior_out, torch.Tensor):
        ior_out = torch.tensor(ior_out, device=ray_d.device)
    
    eta = ior_in / ior_out
    if eta.dim() == 0:
        eta = eta.unsqueeze(0).expand(ray_d.shape[0])
    eta = eta.unsqueeze(-1)  # [N, 1]
    
    # Compute sin^2(theta_t) using Snell's law
    sin2_theta_t = eta**2 * (1.0 - cos_theta_i**2)
    
    # Check for total internal reflection
    valid_mask = (sin2_theta_t <= 1.0).squeeze(-1)
    
    # Compute refracted direction
    cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, min=0.0))
    ray_d_refract = eta * ray_d + (eta * cos_theta_i - cos_theta_t) * normal
    ray_d_refract = F.normalize(ray_d_refract, dim=-1)
    
    return ray_d_refract, valid_mask


def compute_reflection(ray_d, normal):
    """
    Compute reflection direction.
    
    Args:
        ray_d: [N, 3] incident ray direction (normalized)
        normal: [N, 3] surface normal (normalized, pointing outward)
    
    Returns:
        ray_d_reflect: [N, 3] reflected ray direction (normalized)
    """
    ray_d = F.normalize(ray_d, dim=-1)
    normal = F.normalize(normal, dim=-1)
    
    # R = D - 2(D·N)N
    dot_dn = (ray_d * normal).sum(dim=-1, keepdim=True)
    ray_d_reflect = ray_d - 2.0 * dot_dn * normal
    
    return F.normalize(ray_d_reflect, dim=-1)


def compute_fresnel(cos_theta_i, ior_in, ior_out):
    """
    Compute Fresnel reflection coefficient using full Fresnel equations (unpolarized light).
    
    This provides more accurate physical gradients compared to Schlick approximation,
    which is crucial for eliminating geometry ambiguity in ReNeuS.
    
    Args:
        cos_theta_i: [N] cosine of incident angle (absolute value)
        ior_in: [N] or scalar, IOR of incident medium
        ior_out: [N] or scalar, IOR of transmitted medium
    
    Returns:
        fresnel: [N] Fresnel reflection coefficient (0 to 1)
    """
    cos_theta_i = cos_theta_i.abs()
    
    if not isinstance(ior_in, torch.Tensor):
        ior_in = torch.tensor(ior_in, device=cos_theta_i.device)
    if not isinstance(ior_out, torch.Tensor):
        ior_out = torch.tensor(ior_out, device=cos_theta_i.device)
    
    eta = ior_in / ior_out
    if eta.dim() == 0:
        eta = eta.expand(cos_theta_i.shape[0])
    
    # Compute sin^2(theta_t) using Snell's law
    sin2_theta_t = eta**2 * (1.0 - cos_theta_i**2)
    
    # Total internal reflection
    tir_mask = sin2_theta_t > 1.0
    
    cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, min=0.0))
    
    # Fresnel equations for s and p polarization
    # Rs = |( n1*cos(theta_i) - n2*cos(theta_t) ) / ( n1*cos(theta_i) + n2*cos(theta_t) )|^2
    # Rp = |( n2*cos(theta_i) - n1*cos(theta_t) ) / ( n2*cos(theta_i) + n1*cos(theta_t) )|^2
    
    rs_num = ior_in * cos_theta_i - ior_out * cos_theta_t
    rs_den = ior_in * cos_theta_i + ior_out * cos_theta_t
    rs = (rs_num / (rs_den + 1e-7))**2
    
    rp_num = ior_out * cos_theta_i - ior_in * cos_theta_t
    rp_den = ior_out * cos_theta_i + ior_in * cos_theta_t
    rp = (rp_num / (rp_den + 1e-7))**2
    
    # Unpolarized light: average of s and p polarization
    fresnel = 0.5 * (rs + rp)
    
    # Total internal reflection: Fresnel = 1.0
    fresnel = torch.where(tir_mask, torch.ones_like(fresnel), fresnel)
    
    return fresnel


def check_total_internal_reflection(cos_theta_i, ior_in, ior_out):
    """
    Check if total internal reflection (TIR) occurs.
    
    Args:
        cos_theta_i: [N] cosine of incident angle (absolute value)
        ior_in: [N] or scalar, IOR of incident medium
        ior_out: [N] or scalar, IOR of transmitted medium
    
    Returns:
        tir_mask: [N] bool tensor, True if TIR occurs
    """
    cos_theta_i = cos_theta_i.abs()
    
    if not isinstance(ior_in, torch.Tensor):
        ior_in = torch.tensor(ior_in, device=cos_theta_i.device)
    if not isinstance(ior_out, torch.Tensor):
        ior_out = torch.tensor(ior_out, device=cos_theta_i.device)
    
    eta = ior_in / ior_out
    if eta.dim() == 0:
        eta = eta.expand(cos_theta_i.shape[0])
    
    sin2_theta_t = eta**2 * (1.0 - cos_theta_i**2)
    return sin2_theta_t > 1.0


def ray_mesh_intersection(rays_o, rays_d, ray_tracer):
    """
    Batch compute ray-mesh intersections using trimesh ray tracer.
    
    Args:
        rays_o: [N, 3] ray origins
        rays_d: [N, 3] ray directions (should be normalized)
        ray_tracer: trimesh.ray.ray_pyembree.RayMeshIntersector instance
    
    Returns:
        hit_mask: [N] bool tensor, whether each ray hits the mesh
        hit_points: [N, 3] intersection points (0 for missed rays)
        hit_normals: [N, 3] surface normals at intersections (0 for missed rays)
        hit_distances: [N] distances to intersections (inf for missed rays)
    """
    if ray_tracer is None:
        # No container mesh, all rays miss
        batch_size = rays_o.shape[0]
        device = rays_o.device
        return (
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros(batch_size, 3, device=device),
            torch.zeros(batch_size, 3, device=device),
            torch.full((batch_size,), float('inf'), device=device)
        )
    
    # Convert to numpy for trimesh
    rays_o_np = rays_o.detach().cpu().numpy()
    rays_d_np = rays_d.detach().cpu().numpy()
    
    # Perform ray casting
    locations, index_ray, index_tri = ray_tracer.intersects_location(
        rays_o_np, rays_d_np, multiple_hits=False
    )
    
    device = rays_o.device
    batch_size = rays_o.shape[0]
    
    # Initialize outputs
    hit_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    hit_points = torch.zeros(batch_size, 3, device=device)
    hit_normals = torch.zeros(batch_size, 3, device=device)
    hit_distances = torch.full((batch_size,), float('inf'), device=device)
    
    if len(locations) > 0:
        # Get surface normals
        mesh = ray_tracer.mesh
        face_normals = mesh.face_normals[index_tri]
        
        # Convert to torch tensors
        locations_torch = torch.from_numpy(locations).float().to(device)
        normals_torch = torch.from_numpy(face_normals).float().to(device)
        index_ray_torch = torch.from_numpy(index_ray).long()
        
        # Ensure normals point towards the incident ray (for consistent refraction calculation)
        # If ray_d · normal > 0, normal and ray are in the same direction, flip normal
        rays_d_hit = rays_d[index_ray_torch]
        cos_theta = (rays_d_hit * normals_torch).sum(dim=-1, keepdim=True)
        normals_torch = torch.where(cos_theta > 0, -normals_torch, normals_torch)
        
        # Compute distances
        distances = torch.norm(locations_torch - rays_o[index_ray_torch], dim=-1)
        
        # Fill in results
        hit_mask[index_ray_torch] = True
        hit_points[index_ray_torch] = locations_torch
        hit_normals[index_ray_torch] = normals_torch
        hit_distances[index_ray_torch] = distances
    
    return hit_mask, hit_points, hit_normals, hit_distances


# ==============================================================================
# Original NeuS Functions
# ==============================================================================

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 container_mesh_path=None,
                 ior=1.5,
                 max_bounces=3,
                 scale_mat=None):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        
        # [ReNeuS] 新增參數 (NeuS 原版不包含)
        # - ior: 容器折射率 (論文 Sec 4.3)
        # - max_bounces: 遞迴深度 Dre (論文 Sec 3.3, 實驗用 2)
        self.ior = ior
        self.max_bounces = max_bounces
        self.container_mesh = None  # [ReNeuS] NeuS 原版無容器 mesh
        self.ray_tracer = None  # [ReNeuS] trimesh ray tracer 實例
        self.scale_mat = scale_mat  # [ReNeuS] for mesh normalization
        
        if container_mesh_path is not None:
            try:
                import trimesh
                logging.info(f'[ReNeuS] Loading container mesh from: {container_mesh_path}')
                self.container_mesh = trimesh.load(container_mesh_path, force='mesh')
                
                # Apply scale_mat inversion to align mesh with Normalized Space
                if self.scale_mat is not None:
                    scale_mat_inv = np.linalg.inv(self.scale_mat)
                    self.container_mesh.apply_transform(scale_mat_inv)
                    logging.info('[ReNeuS] Applied scale_mat_inv to container_mesh (world -> norm)')
                
                # Initialize ray tracer with Embree acceleration (requires pyembree)
                try:
                    self.ray_tracer = trimesh.ray.ray_pyembree.RayMeshIntersector(
                        self.container_mesh
                    )
                    logging.info('[ReNeuS] Using Embree ray tracer (accelerated)')
                except:
                    logging.warning('[ReNeuS] Embree not available, using default ray tracer (slower)')
                    self.ray_tracer = trimesh.ray.ray_triangle.RayMeshIntersector(
                        self.container_mesh
                    )
                
                logging.info(f'[ReNeuS] Container mesh loaded: {self.container_mesh.faces.shape[0]} faces, IOR={self.ior}')
            except Exception as e:
                logging.error(f'[ReNeuS] Failed to load container mesh: {e}')
                self.container_mesh = None
                self.ray_tracer = None


    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        Not used in ReNeuS
        """
        raise NotImplementedError
        # batch_size, n_samples = z_vals.shape

        # # Section length
        # dists = z_vals[..., 1:] - z_vals[..., :-1]
        # dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        # mid_z_vals = z_vals + dists * 0.5

        # # Section midpoints
        # pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        # dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        # pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        # dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        # pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        # dirs = dirs.reshape(-1, 3)

        # density, sampled_color = nerf(pts, dirs)
        # sampled_color = torch.sigmoid(sampled_color)
        # alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        # alpha = alpha.reshape(batch_size, n_samples)
        # weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        # color = (weights[:, :, None] * sampled_color).sum(dim=1)
        # if background_rgb is not None:
        #     color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        # return {
        #     'color': color,
        #     'sampled_color': sampled_color,
        #     'alpha': alpha,
        #     'weights': weights,
        # }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        # [ReNeuS] 容器內所有採樣點都有效，不需 unit sphere 判定
        inside_sphere = torch.ones(batch_size, n_samples - 1, dtype=torch.bool, device=z_vals.device)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        # [ReNeuS] 容器內所有採樣點都有效，不需 unit sphere 判定
        # NeuS 原版用 inside_sphere 區分 SDF/NeRF++ 區域，ReNeuS 不需要 (n_outside=0)
        inside_sphere = torch.ones(batch_size, n_samples, device=pts.device)
        relax_inside_sphere = inside_sphere

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, extract_inner_render=False):
        """
        [ReNeuS] Deterministic Branching 渲染 (論文 Sec 3.3, Eq.7-12)

        NeuS 原版的 render() 方法僅做單次體渲染。
        ReNeuS 新增本方法以實現論文的 "hybrid rendering strategy"：
        
        核心差異（vs stochastic sampling）：
        - 論文 Eq.8: C(ℓ) = R · C(ℓ_r) + T_re · C(ℓ_t)
          在每個界面**同時計算**反射+折射兩條路徑，並以 Fresnel 係數加權累積。
        - 論文 Eq.10: C(ℓ_i) = ACC_{m_i ∈ L_i}(C(m_i))
          累積**所有** sub-ray 的顏色貢獻。
        
        實現方式：
        - 使用 pixel_idx 追蹤多條 sub-ray 對應到同一原始像素
        - 每個界面產生反射+折射兩組 ray（權重分別為 R 和 1-R）
        - 使用 scatter_add_ 將所有 sub-ray 的顏色累積回原始像素
        - Dre=2 時每像素最多 4 條 sub-ray
        
        其他功能：
        - 全內反射 (TIR) 處理
        - Throughput 累積 (Eq.11: T = exp(-∫ρds))
        - Gamma Correction (Eq.12: I = C^(1/2.2))
        - 固定背景色 C_out (Sec 4.3)

        與 NeuS 原版 render() 的結構對照：
        ┌─────────────────────────────────────────────────────────┐
        │ NeuS render()              │ ReNeuS render()            │
        ├────────────────────────────┼────────────────────────────┤
        │ 1. z_vals (near→far)       │ 1. Iterative ray tracing   │
        │ 2. Perturb                 │    with branching (Eq.8)   │
        │ 3. Up-sample (importance)  │    - Container intersection│
        │ 4. Background model        │    - Volume render inside  │
        │ 5. render_core             │    - Fresnel split         │
        │ 6. Pack & return           │ 2. Background (fixed 0.8)  │
        │                            │ 3. Gamma correction (Eq.12)│
        │                            │ 4. Pack & return           │
        └────────────────────────────┴────────────────────────────┘
        """
        device = rays_o.device
        batch_size = len(rays_o)

        inner_color_fine = None
        if extract_inner_render:
            with torch.no_grad():
                # Bypass container intersection and render SDF network directly
                z_vals_in = torch.linspace(0.0, 1.0, self.n_samples, device=device)
                z_vals_in = near + (far - near) * z_vals_in[None, :]

                perturb_in = self.perturb if perturb_overwrite < 0 else perturb_overwrite
                if perturb_in > 0:
                    t_rand_in = (torch.rand([batch_size, 1], device=device) - 0.5)
                    z_vals_in = z_vals_in + t_rand_in * (far - near) / self.n_samples

                if self.n_importance > 0:
                    pts_in = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_in[..., :, None]
                    sdf_in = self.sdf_network.sdf(pts_in.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                    for i in range(self.up_sample_steps):
                        new_z_vals_in = self.up_sample(rays_o, rays_d, z_vals_in, sdf_in,
                                                    self.n_importance // self.up_sample_steps, 64 * 2**i)
                        z_vals_in, sdf_in = self.cat_z_vals(rays_o, rays_d, z_vals_in, new_z_vals_in, sdf_in,
                                                      last=(i + 1 == self.up_sample_steps))

                sample_dist_in = ((far - near) / self.n_samples).mean().item()
                ret_in = self.render_core(
                    rays_o, rays_d, z_vals_in, sample_dist_in,
                    self.sdf_network, self.deviation_network, self.color_network,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=cos_anneal_ratio
                )

                inner_color_fine = torch.clamp(ret_in['color'], 0.0, 1.0)
                inner_color_fine = torch.pow(inner_color_fine + 1e-8, 1.0 / 2.2)

        # =====================================================================
        # Output accumulators (fixed size, indexed by original pixel)
        # Corresponds to NeuS: color_fine, s_val, weights_sum, gradients, etc.
        # =====================================================================
        final_color = torch.zeros(batch_size, 3, device=device)
        ret_s_val = torch.zeros(batch_size, 1, device=device)
        ret_weights_sum = torch.zeros(batch_size, 1, device=device)
        ret_gradient_error = torch.tensor(0.0, device=device)
        ret_gradients = torch.zeros(batch_size, self.n_samples, 3, device=device)
        ret_weights = torch.zeros(batch_size, self.n_samples, device=device)
        ret_inside_sphere = torch.zeros(batch_size, self.n_samples, device=device)
        # [ReNeuS Eq.15] Transmittance loss 累積器：sum over all sub-rays of ||1-T_ℓ||
        ret_trans_loss = torch.tensor(0.0, device=device)
        has_rendered_volume = False

        # =====================================================================
        # Active ray state (variable size, grows with branching)
        # pixel_idx: which original pixel each active sub-ray belongs to
        # =====================================================================
        pixel_idx = torch.arange(batch_size, device=device)
        curr_o = rays_o.clone()
        curr_d = rays_d.clone()
        curr_ior = torch.ones(batch_size, device=device)          # 1.0 = Air
        curr_weight = torch.ones(batch_size, 1, device=device)    # Cumulative Fresnel weight

        # =====================================================================
        # Iterative ray tracing with deterministic branching
        # NeuS: single render_core call. ReNeuS: loop over bounces.
        # =====================================================================
        for bounce in range(self.max_bounces):
            n_active = len(pixel_idx)
            if n_active == 0:
                break

            # ----- Step 1: Intersect with Container Mesh -----
            hit_mask, hit_points, hit_normals, hit_distances = ray_mesh_intersection(
                curr_o, curr_d, self.ray_tracer
            )

            # ----- Step 2: Handle Misses → background -----
            miss_mask = ~hit_mask
            if miss_mask.any() and background_rgb is not None:
                # [ReNeuS Sec 4.3] 固定背景色 C_out
                bg_contrib = curr_weight[miss_mask] * background_rgb
                final_color.scatter_add_(
                    0, pixel_idx[miss_mask].unsqueeze(-1).expand(-1, 3), bg_contrib
                )

            # Filter to hitting rays only, removing miss rays from active set
            # (miss rays already got background above, must not be processed again)
            if not hit_mask.any():
                # All rays missed → clear active set so post-loop won't double-add bg
                pixel_idx = pixel_idx[:0]
                break

            h_idx = torch.where(hit_mask)[0]
            h_pixel = pixel_idx[h_idx]
            h_o = curr_o[h_idx]
            h_d = curr_d[h_idx]
            h_ior = curr_ior[h_idx]
            h_weight = curr_weight[h_idx]
            h_pts = hit_points[h_idx]
            h_norms = hit_normals[h_idx]
            h_dists = hit_distances[h_idx]

            # ----- Step 3: Volume Rendering inside container -----
            # NeuS: single render_core with z_vals from near→far
            # ReNeuS: render_core for rays INSIDE container (IOR > 1)
            is_inside = (h_ior > 1.0)

            if is_inside.any():
                inside_sel = torch.where(is_inside)[0]
                batch_active = len(inside_sel)

                dist_to_exit = h_dists[inside_sel]
                rays_o_active = h_o[inside_sel]
                rays_d_active = h_d[inside_sel]

                # --- z_vals (NeuS: near→far, ReNeuS: origin→exit) ---
                z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
                z_vals = z_vals[None, :].expand(batch_active, -1) * dist_to_exit[:, None]

                # --- Perturb (same as NeuS) ---
                perturb = self.perturb if perturb_overwrite < 0 else perturb_overwrite
                if perturb > 0:
                    t_rand = (torch.rand([batch_active, 1], device=device) - 0.5)
                    z_vals = z_vals + t_rand * (dist_to_exit[:, None] / self.n_samples)

                # --- Up-sample / importance sampling (same as NeuS) ---
                if self.n_importance > 0:
                    with torch.no_grad():
                        pts = rays_o_active[:, None, :] + rays_d_active[:, None, :] * z_vals[..., :, None]
                        sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_active, self.n_samples)

                        for i in range(self.up_sample_steps):
                            new_z_vals = self.up_sample(
                                rays_o_active, rays_d_active, z_vals, sdf,
                                self.n_importance // self.up_sample_steps, 64 * 2**i
                            )
                            z_vals, sdf = self.cat_z_vals(
                                rays_o_active, rays_d_active, z_vals, new_z_vals, sdf,
                                last=(i + 1 == self.up_sample_steps)
                            )

                # --- Render core (same as NeuS) ---
                sample_dist = dist_to_exit.mean().item() / self.n_samples

                ret_fine = self.render_core(
                    rays_o_active, rays_d_active, z_vals, sample_dist,
                    self.sdf_network, self.deviation_network, self.color_network,
                    background_rgb=None,
                    cos_anneal_ratio=cos_anneal_ratio
                )

                vol_color = ret_fine['color']
                vol_weights_sum = ret_fine['weights'].sum(dim=-1, keepdim=True)

                # --- Accumulate (NeuS: direct assign. ReNeuS: scatter_add_) ---
                contribution = h_weight[inside_sel] * vol_color
                inside_pixels = h_pixel[inside_sel]
                final_color.scatter_add_(
                    0, inside_pixels.unsqueeze(-1).expand(-1, 3), contribution
                )

                # [ReNeuS Eq.15] Transmittance loss: 累加 ||1-T_ℓ|| (不取 mean，由 exp_runner 做 /|M_in|)
                ret_trans_loss = ret_trans_loss + vol_weights_sum.sum()

                # [ReNeuS Eq.11] Update throughput: T_new = T_old * (1 - opacity)
                h_weight[inside_sel] = h_weight[inside_sel] * (1.0 - vol_weights_sum).clamp(0.0, 1.0)

                # --- Store metrics from first volume render ---
                # NeuS: directly from ret_fine. ReNeuS: scatter to original pixels.
                if not has_rendered_volume:
                    ret_gradient_error = ret_fine['gradient_error']
                    orig_pixels = inside_pixels
                    ret_s_val[orig_pixels] = ret_fine['s_val'].reshape(batch_active, -1).mean(dim=-1, keepdim=True)
                    ret_weights_sum[orig_pixels] = vol_weights_sum

                    feat_n_samples = ret_fine['gradients'].shape[1]
                    if feat_n_samples >= self.n_samples:
                        ret_gradients[orig_pixels] = ret_fine['gradients'][:, :self.n_samples, :]
                        ret_weights[orig_pixels] = ret_fine['weights'][:, :self.n_samples]
                        ret_inside_sphere[orig_pixels] = ret_fine['inside_sphere'][:, :self.n_samples]

                    has_rendered_volume = True

            # ------------------------------------------------------------------
            # Step 4: Deterministic Branching at Interface (Eq.8-9)
            # ------------------------------------------------------------------
            # [ReNeuS Eq.8] C(ℓ) = R · C(ℓ_r) + T_re · C(ℓ_t)
            # 在每個界面同時產生反射+折射兩組 ray

            # Determine IOR transition
            ior1 = h_ior
            ior2 = torch.where(
                ior1 == 1.0,
                torch.full_like(ior1, self.ior),
                torch.ones_like(ior1)
            )

            # Compute Fresnel coefficient R (Eq.8-9, natural light assumption)
            cos_theta_i = torch.abs((h_d * h_norms).sum(dim=-1))
            fresnel_R = compute_fresnel(cos_theta_i, ior1, ior2)

            # Compute reflection direction (always valid)
            d_reflect = compute_reflection(h_d, h_norms)

            # Compute refraction direction (may fail for TIR)
            d_refract, valid_refract = compute_refraction(
                h_d, h_norms, ior1, ior2
            )

            # TIR: Fresnel R = 1.0 (full reflection, no refraction branch)
            tir_mask = ~valid_refract
            fresnel_R = torch.where(tir_mask, torch.ones_like(fresnel_R), fresnel_R)
            fresnel_T = 1.0 - fresnel_R

            # Reflection branch
            reflect_o = h_pts + 1e-4 * d_reflect
            reflect_ior = ior1

            # Refraction branch: only non-TIR rays, weight *= T
            refract_valid = valid_refract
            refract_o = h_pts[refract_valid] + 1e-4 * d_refract[refract_valid]
            refract_ior = ior2[refract_valid]

            # Concatenate both branches
            next_pixel_idx = torch.cat([h_pixel, h_pixel[refract_valid]])
            next_o = torch.cat([reflect_o, refract_o])
            next_d = torch.cat([d_reflect, d_refract[refract_valid]])
            next_ior = torch.cat([reflect_ior, refract_ior])
            next_weight = torch.cat([
                h_weight * fresnel_R.unsqueeze(-1),
                h_weight[refract_valid] * fresnel_T[refract_valid].unsqueeze(-1)
            ])

            # Update active ray state for next iteration
            pixel_idx = next_pixel_idx
            curr_o = next_o
            curr_d = next_d
            curr_ior = next_ior
            curr_weight = next_weight

        # =====================================================================
        # End of loop: assign background to remaining active rays
        # NeuS: background via NeRF++ or fixed color in render_core
        # ReNeuS: fixed background C_out (Sec 4.3)
        # =====================================================================
        if len(pixel_idx) > 0 and background_rgb is not None:
            bg_contrib = curr_weight * background_rgb
            final_color.scatter_add_(
                0, pixel_idx.unsqueeze(-1).expand(-1, 3), bg_contrib
            )

        # Gradient flow protection
        if not has_rendered_volume:
            dummy_sdf = self.sdf_network.sdf(rays_o[:1])
            final_color = final_color + dummy_sdf.sum() * 0.0
            ret_gradient_error = torch.tensor(0.01, device=device, requires_grad=True)

        # [ReNeuS Eq.12] Gamma Correction: I = C^(1/2.2)
        final_color = torch.clamp(final_color, 0.0, 1.0)
        final_color = torch.pow(final_color + 1e-8, 1.0 / 2.2)

        # =====================================================================
        # Pack & return (same keys as NeuS + trans_loss)
        # =====================================================================
        ret_dict = {
            'color_fine': final_color,
            's_val': ret_s_val,
            'cdf_fine': torch.zeros(batch_size, self.n_samples, device=device),
            'weight_sum': ret_weights_sum,
            'weight_max': ret_weights_sum,
            'gradients': ret_gradients,
            'weights': ret_weights,
            'gradient_error': ret_gradient_error,
            'inside_sphere': ret_inside_sphere,
            'trans_loss': ret_trans_loss,  # [ReNeuS Eq.15] (NeuS 無此項)
        }
        
        if inner_color_fine is not None:
            ret_dict['inner_color_fine'] = inner_color_fine
            
        return ret_dict

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
