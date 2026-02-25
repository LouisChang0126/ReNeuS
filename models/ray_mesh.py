"""GPU Ray-Mesh Intersection using Möller–Trumbore algorithm.

Pure PyTorch implementation for GPU-accelerated ray-triangle intersection.
No external dependencies beyond PyTorch and trimesh.

Ported from 3dgrut-R/threedgrut/refraction/ray_mesh.py
"""

import torch
import trimesh


# =============================================================================
# Constants
# =============================================================================

EPSILON = 1e-6


# =============================================================================
# GPU Ray-Mesh Intersector
# =============================================================================

class GPURayMeshIntersector:
    """GPU-accelerated ray-mesh intersection using Möller–Trumbore algorithm.

    Precomputes triangle vertex positions on GPU for efficient batch
    ray-triangle intersection queries.

    Drop-in replacement for trimesh's CPU-based RayMeshIntersector in ReNeuS.
    The intersect_batch() return signature matches ray_mesh_intersection():
        (hit_mask, hit_points, hit_normals, hit_distances)
    """

    def __init__(self, mesh, device='cuda'):
        """
        Initialize GPU ray-mesh intersector.

        Args:
            mesh (trimesh.Trimesh): Mesh object.
            device (str): CUDA device string.
        """
        self.device = torch.device(device)

        # Move mesh data to GPU
        self.vertices = torch.tensor(
            mesh.vertices, dtype=torch.float32, device=self.device
        )
        self.faces = torch.tensor(
            mesh.faces, dtype=torch.long, device=self.device
        )
        self.face_normals = torch.tensor(
            mesh.face_normals, dtype=torch.float32, device=self.device
        )

        # Precompute triangle vertices
        self.v0 = self.vertices[self.faces[:, 0]]  # (F, 3)
        self.v1 = self.vertices[self.faces[:, 1]]  # (F, 3)
        self.v2 = self.vertices[self.faces[:, 2]]  # (F, 3)
        self.n_faces = len(self.faces)

    def intersect(self, ray_o, ray_d):
        """
        Find closest intersection with mesh for a single ray.

        Args:
            ray_o (torch.Tensor): Ray origin (3,).
            ray_d (torch.Tensor): Ray direction (3,), normalized.

        Returns:
            hit (bool): Whether ray hits mesh.
            point (torch.Tensor): Intersection point (3,).
            normal (torch.Tensor): Surface normal at intersection (3,).
            t (float): Distance along ray.
        """
        # Expand ray for all triangles: (F, 3)
        ray_o_exp = ray_o.view(1, 3).expand(self.n_faces, 3)
        ray_d_exp = ray_d.view(1, 3).expand(self.n_faces, 3)

        # Möller–Trumbore algorithm
        edge1 = self.v1 - self.v0  # (F, 3)
        edge2 = self.v2 - self.v0  # (F, 3)

        h = torch.cross(ray_d_exp, edge2, dim=1)  # (F, 3)
        a = (edge1 * h).sum(dim=1)  # (F,)

        # Filter parallel rays
        valid = torch.abs(a) > EPSILON

        f = torch.where(valid, 1.0 / a, torch.zeros_like(a))
        s = ray_o_exp - self.v0
        u = f * (s * h).sum(dim=1)

        # Check u bounds
        valid = valid & (u >= 0) & (u <= 1)

        q = torch.cross(s, edge1, dim=1)
        v = f * (ray_d_exp * q).sum(dim=1)

        # Check v bounds
        valid = valid & (v >= 0) & (u + v <= 1)

        t = f * (edge2 * q).sum(dim=1)

        # Check t > 0 (in front of ray)
        valid = valid & (t > EPSILON)

        if not valid.any():
            zero = torch.zeros(3, device=self.device)
            return False, zero, zero, 0.0

        # Find closest hit
        t_valid = torch.where(valid, t, torch.full_like(t, float('inf')))
        min_idx = t_valid.argmin()

        t_hit = t[min_idx].item()
        point = ray_o + ray_d * t_hit
        normal = self.face_normals[min_idx]

        return True, point, normal, t_hit

    def intersect_batch(self, rays_o, rays_d):
        """
        Find closest intersection for multiple rays (batch).

        Return signature matches the legacy ray_mesh_intersection() function:
            (hit_mask, hit_points, hit_normals, hit_distances)

        Args:
            rays_o (torch.Tensor): Ray origins (N, 3).
            rays_d (torch.Tensor): Ray directions (N, 3), normalized.

        Returns:
            hit (torch.Tensor): Whether each ray hits mesh (N,), bool.
            points (torch.Tensor): Intersection points (N, 3).
            normals (torch.Tensor): Surface normals at hit points (N, 3).
            t (torch.Tensor): Closest hit distances (N,), inf if no hit.
        """
        N = rays_o.shape[0]
        F = self.n_faces

        # Expand for all ray-triangle combinations: (N, F, 3)
        rays_o_exp = rays_o.unsqueeze(1).expand(N, F, 3)
        rays_d_exp = rays_d.unsqueeze(1).expand(N, F, 3)
        v0_exp = self.v0.unsqueeze(0).expand(N, F, 3)
        v1_exp = self.v1.unsqueeze(0).expand(N, F, 3)
        v2_exp = self.v2.unsqueeze(0).expand(N, F, 3)

        # Möller–Trumbore (vectorized)
        edge1 = v1_exp - v0_exp  # (N, F, 3)
        edge2 = v2_exp - v0_exp  # (N, F, 3)

        h = torch.cross(rays_d_exp, edge2, dim=2)  # (N, F, 3)
        a = (edge1 * h).sum(dim=2)  # (N, F)

        valid = torch.abs(a) > EPSILON
        f = torch.where(valid, 1.0 / a, torch.zeros_like(a))
        s = rays_o_exp - v0_exp
        u = f * (s * h).sum(dim=2)

        valid = valid & (u >= 0) & (u <= 1)

        q = torch.cross(s, edge1, dim=2)
        v = f * (rays_d_exp * q).sum(dim=2)

        valid = valid & (v >= 0) & (u + v <= 1)

        t = f * (edge2 * q).sum(dim=2)
        valid = valid & (t > EPSILON)

        # Find closest hit per ray
        t_valid = torch.where(valid, t, torch.full_like(t, float('inf')))
        min_t, min_idx = t_valid.min(dim=1)  # (N,), (N,)

        hit = min_t < float('inf')
        points = rays_o + rays_d * min_t.unsqueeze(-1)
        normals = self.face_normals[min_idx]  # (N, 3)

        # Ensure normals point opposite to ray direction (consistent with
        # the legacy ray_mesh_intersection() convention used in renderer.py)
        cos_theta = (rays_d * normals).sum(dim=-1, keepdim=True)
        normals = torch.where(cos_theta > 0, -normals, normals)

        return hit, points, normals, min_t
