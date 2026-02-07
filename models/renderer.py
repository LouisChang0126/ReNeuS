import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic


# ==============================================================================
# ReNeuS: Ray Tracing Utility Functions
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
                 max_bounces=3):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        
        # ReNeuS extensions
        self.ior = ior
        self.max_bounces = max_bounces
        self.container_mesh = None
        self.ray_tracer = None
        
        if container_mesh_path is not None:
            try:
                import trimesh
                logging.info(f'[ReNeuS] Loading container mesh from: {container_mesh_path}')
                self.container_mesh = trimesh.load(container_mesh_path, force='mesh')
                
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
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
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
                    background_alpha=None,
                    background_sampled_color=None,
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

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

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

    def render_with_refraction(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        """
        ReNeuS rendering with refraction-aware ray tracing.
        Simplified version: single refraction entry into container, then standard NeuS rendering.
        """
        batch_size = rays_o.shape[0]
        device = rays_o.device
        
        # Step 1: Compute intersection with container surface
        hit_mask, hit_points, hit_normals, hit_distances = ray_mesh_intersection(
            rays_o, rays_d, self.ray_tracer
        )
        
        # Initialize output
        color_fine = torch.zeros(batch_size, 3, device=device)
        s_val = torch.zeros(batch_size, 1, device=device)
        weight_sum = torch.zeros(batch_size, 1, device=device)
        
        # Step 2: Handle rays that miss the container
        if background_rgb is not None:
            color_fine[~hit_mask] = background_rgb
        
        if not hit_mask.any():
            return {
                'color_fine': color_fine,
                's_val': s_val,
                'weight_sum': weight_sum,
                'gradient_error': torch.tensor(0.0, device=device),
                'weights': torch.zeros(batch_size, self.n_samples, device=device),
                'gradients': torch.zeros(batch_size, self.n_samples, 3, device=device)
            }
        
        # Step 3: Compute refraction for hitting rays
        hit_indices = torch.where(hit_mask)[0]
        rays_d_refract, _ = compute_refraction(
            rays_d[hit_indices], hit_normals[hit_indices], ior_in=1.0, ior_out=self.ior
        )
        
        # Offset to avoid self-intersection
        rays_o_refract = hit_points[hit_indices] + 1e-4 * rays_d_refract
        
        # Step 4: Render along refracted rays
        batch_size_hit = hit_indices.shape[0]
        sample_dist = 2.0 / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        z_vals = z_vals[None, :].expand(batch_size_hit, -1) * 2.0  # Simple far=2.0
        
        perturb = self.perturb if perturb_overwrite < 0 else perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size_hit, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples
        
        # Up-sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o_refract[:, None, :] + rays_d_refract[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size_hit, self.n_samples)
                
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(
                        rays_o_refract, rays_d_refract, z_vals, sdf,
                        self.n_importance // self.up_sample_steps, 64 * 2**i
                    )
                    z_vals, sdf = self.cat_z_vals(
                        rays_o_refract, rays_d_refract, z_vals, new_z_vals, sdf,
                        last=(i + 1 == self.up_sample_steps)
                    )
        
        # Render core
        ret_fine = self.render_core(
            rays_o_refract, rays_d_refract, z_vals, sample_dist,
            self.sdf_network, self.deviation_network, self.color_network,
            background_rgb=background_rgb, cos_anneal_ratio=cos_anneal_ratio
        )
        
        # Fill results
        color_fine[hit_indices] = ret_fine['color']
        weight_sum[hit_indices] = ret_fine['weights'].sum(dim=-1, keepdim=True)
        
        n_samples_final = z_vals.shape[1]
        s_val_mean = ret_fine['s_val'].reshape(batch_size_hit, n_samples_final).mean(dim=-1, keepdim=True)
        s_val[hit_indices] = s_val_mean
        
        gradients_all = torch.zeros(batch_size, n_samples_final, 3, device=device)
        gradients_all[hit_indices] = ret_fine['gradients']
        
        weights_all = torch.zeros(batch_size, n_samples_final, device=device)
        weights_all[hit_indices] = ret_fine['weights']
        
        return {
            'color_fine': color_fine,
            's_val': s_val,
            'weight_sum': weight_sum,
            'weight_max': weight_sum,
            'gradient_error': ret_fine['gradient_error'],
            'gradients': gradients_all,
            'weights': weights_all,
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        # ReNeuS: If container mesh is loaded, apply refraction-aware rendering
        if self.container_mesh is not None:
            return self.render_with_refraction(
                rays_o, rays_d, near, far, perturb_overwrite, background_rgb, cos_anneal_ratio
            )
        
        # Original NeuS rendering (backward compatible)
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
