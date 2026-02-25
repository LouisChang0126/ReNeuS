"""
ReNeuS Dataset — 3DGRUT / NeRF transforms.json format
=======================================================
Reads camera poses and images from transforms_<split>.json (3DGRUT convention).

Image format:
  - RGBA PNG: RGB = colour,  A = object mask (alpha > 0 → foreground)

Camera parameters:
  - fl_x, fl_y, cx, cy, w, h in the JSON top-level
  - transform_matrix: 4x4 camera-to-world (same convention as the original ReNeuS npz)

Normalization:
  - glass box PLY (mesh_outside) is used to compute a scene centre + bounding sphere,
    and the resulting scale_mat is used for near/far estimation and bbox extraction.
"""

import json

import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
import trimesh
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# ──────────────────────────────────────────────────────────────────────────────
# Normalization helper
# ──────────────────────────────────────────────────────────────────────────────

def _compute_scale_mat(glass_ply_path: str, scale_mat_scale: float = 1.1) -> np.ndarray:
    """
    Build a 4x4 scale_mat from the glass container PLY so the scene fits
    inside the unit sphere.

    Convention (same as cameras_sphere.npz):
        x_world = scale_mat @ [x_norm; 1]
        s = bounding_radius * scale_mat_scale
    """
    mesh   = trimesh.load(glass_ply_path, force='mesh')
    verts  = mesh.vertices
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    radius = np.linalg.norm(verts - center, axis=1).max()
    s      = float(radius * scale_mat_scale)
    mat = np.array([
        [s, 0, 0, center[0]],
        [0, s, 0, center[1]],
        [0, 0, s, center[2]],
        [0, 0, 0, 1.0],
    ], dtype=np.float32)
    return mat


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class Dataset:
    def __init__(self, conf):
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir        = conf.get_string('data_dir')
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        train_split          = conf.get_string('train_split',    default='train')

        # ── Load JSON ──────────────────────────────────────────────────────────
        json_path = os.path.join(self.data_dir, f'transforms_{train_split}.json')
        with open(json_path, 'r') as f:
            meta = json.load(f)

        # ── Intrinsics ────────────────────────────────────────────────────────
        W  = int(meta['w'])
        H  = int(meta['h'])
        fx = float(meta['fl_x'])
        fy = float(meta['fl_y'])
        cx = float(meta['cx'])
        cy = float(meta['cy'])

        K = np.array([
            [fx,  0, cx, 0],
            [ 0, fy, cy, 0],
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1],
        ], dtype=np.float32)

        # ── ReNeuS metadata ───────────────────────────────────────────────────
        self.ior                 = meta.get('IOR', None)
        glass_rel                = meta.get('mesh_outside', None)
        object_rel               = meta.get('mesh_inside',  None)
        self.container_mesh_path = os.path.join(self.data_dir, glass_rel)  if glass_rel  else None
        self.object_mesh_path    = os.path.join(self.data_dir, object_rel) if object_rel else None
        print(f'[ReNeuS] Loaded metadata: IOR={self.ior}, container_mesh={self.container_mesh_path}')

        # ── Normalization (scale_mat) ──────────────────────────────────────────
        if self.container_mesh_path and os.path.exists(self.container_mesh_path):
            scale_mat    = _compute_scale_mat(self.container_mesh_path, self.scale_mat_scale)
            scale_mat_inv = np.linalg.inv(scale_mat)
        else:
            print('[ReNeuS][warn] glass mesh not found, using identity scale_mat')
            scale_mat     = np.eye(4, dtype=np.float32)
            scale_mat_inv = np.eye(4, dtype=np.float32)
        self.scale_mat = scale_mat   # kept for object_bbox computation

        # ── Load frames ───────────────────────────────────────────────────────
        frames = meta['frames']
        self.n_images = len(frames)

        images_list = []
        masks_list  = []
        poses_list  = []

        for frame in frames:
            # ---- pose: NeRF/OpenGL c2w → Normalized Space OpenCV w2c ----
            # Step 1: OpenGL c2w → OpenCV w2c (flip Y & Z)
            c2w_nerf = np.array(frame['transform_matrix'], dtype=np.float32)
            flip     = np.diag([1., -1., -1., 1.]).astype(np.float32)
            w2c_cv   = flip @ np.linalg.inv(c2w_nerf)  # world-to-camera (OpenCV)

            # Step 2: Fold scale_mat into projection → RQ decompose → normalized pose
            # P_norm = K @ w2c_cv @ scale_mat  maps normalized coords → image
            # cv.decomposeProjectionMatrix recovers (K_new, R_w2c, cam_center)
            P_norm = (K @ w2c_cv @ scale_mat)[:3, :4]
            _, R_dec, t_dec = cv.decomposeProjectionMatrix(P_norm)[:3]
            cam_center = (t_dec[:3] / t_dec[3])[:, 0]  # camera position in norm space

            c2w_norm = np.eye(4, dtype=np.float32)
            c2w_norm[:3, :3] = R_dec.T            # R_dec is w2c rotation
            c2w_norm[:3, 3]  = cam_center
            w2c_norm = np.linalg.inv(c2w_norm).astype(np.float32)

            poses_list.append(w2c_norm)

            # ---- image path ---------------------------------------------------
            fp = frame['file_path']
            img_path = os.path.join(self.data_dir, fp)
            if not os.path.exists(img_path):
                img_path = img_path + '.png'
            if not os.path.exists(img_path):
                raise FileNotFoundError(f'Image not found: {img_path}')

            # ---- load RGBA image ---------------------------------------------
            # Use cv2 with IMREAD_UNCHANGED to keep alpha channel
            img_bgra = cv.imread(img_path, cv.IMREAD_UNCHANGED)  # (H, W, 4) BGR+A
            if img_bgra is None:
                raise IOError(f'Cannot read image: {img_path}')

            # 確認大小一致
            assert img_bgra.shape[:2] == (H, W), \
                f'Image size mismatch: {img_bgra.shape[:2]} vs ({H},{W})'

            if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
                # RGBA stored as BGRA by OpenCV (may be uint8 or uint16)
                scale   = 65535.0 if img_bgra.dtype == np.uint16 else 255.0
                alpha   = img_bgra[:, :, 3].astype(np.float32) / scale  # (H,W) [0,1]
                mask    = (alpha > 0).astype(np.float32)
                # Zero out RGB where alpha==0 to avoid background colour leaking
                # (ref: visualize_alignment.py: "Zero out RGB in transparent areas")
                img_rgb = (img_bgra[:, :, 2::-1].astype(np.float32) / scale)  # BGR->RGB [0,1]
                img_rgb = img_rgb * mask[:, :, None]  # black background for masked pixels
            else:
                scale   = 65535.0 if img_bgra.dtype == np.uint16 else 255.0
                img_rgb = (img_bgra[:, :, :3][:, :, ::-1].astype(np.float32) / scale)
                mask    = np.ones((H, W), dtype=np.float32)

            images_list.append(img_rgb)                                   # (H,W,3) float32 [0,1]
            # Expand mask to 3 channels to match existing code that does mask[:, :1]
            masks_list.append(np.repeat(mask[:, :, None], 3, axis=2))    # (H,W,3) float32

        # Assemble tensors
        images_np = np.stack(images_list, axis=0)   # (N, H, W, 3)
        masks_np  = np.stack(masks_list,  axis=0)   # (N, H, W, 3)
        poses_np  = np.stack(poses_list,  axis=0)   # (N, 4, 4) -- OpenCV w2c

        self.images = torch.from_numpy(images_np).float().cpu()   # (N,H,W,3)
        self.masks  = torch.from_numpy(masks_np ).float().cpu()   # (N,H,W,3)
        self.H, self.W = H, W
        self.image_pixels = H * W

        # Intrinsics: same K for all cameras (OpenCV convention: right, down, front)
        K_t = torch.from_numpy(K).float()
        self.intrinsics_all     = K_t.unsqueeze(0).repeat(self.n_images, 1, 1).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]

        # pose_all: Normalized-Space OpenCV w2c, (N,4,4) on CUDA
        # scale_mat has been folded in: rays are in [-1,1]^3 centred at origin.
        self.pose_all = torch.from_numpy(poses_np).float().to(self.device)  # (N,4,4)

        # Object bounding box — already in normalized space (unit cube)
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        # Keep scale_mats_np list for near_far_from_sphere / any legacy code
        self.scale_mats_np = [scale_mat] * self.n_images

        print(f'Load data: End  (n_images={self.n_images}, H={H}, W={W}, IOR={self.ior})')

    # ──────────────────────────────────────────────────────────────────────────
    # Ray generation
    # ──────────────────────────────────────────────────────────────────────────

    def gen_rays_at(self, img_idx, resolution_level=1):
        """Generate rays for a full image (for validation).

        pose_all stores OpenCV world-to-camera (w2c):
          - camera position = R^T @ (-t)   where R = w2c[:3,:3], t = w2c[:3,3]
          - ray direction   = R^T @ K_inv @ [px, py, 1]^T   (OpenCV: +z is forward)
        """
        l  = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l, device=self.device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=self.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # (W,H,3)
        # Unproject: direction in camera space (OpenCV, +z forward)
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, None, :3, :3],
            p[:, :, :, None]
        ).squeeze()   # (W,H,3)
        # Rotate to world space: R^T @ p_cam   (R = w2c[:3,:3])
        R = self.pose_all[img_idx, :3, :3]  # (3,3)  w2c rotation
        rays_v = torch.matmul(R.T[None, None], p[:, :, :, None]).squeeze()  # (W,H,3)
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)
        # Camera origin: R^T @ (-t)
        t = self.pose_all[img_idx, :3, 3]   # (3,)  w2c translation
        rays_o = (R.T @ (-t)).expand(rays_v.shape)  # (W,H,3)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """Generate random rays for training.

        pose_all stores OpenCV world-to-camera (w2c).
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size], device=self.device)
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size], device=self.device)
        color = self.images[img_idx][(pixels_y.cpu(), pixels_x.cpu())].to(self.device)  # (B,3)
        mask  = self.masks [img_idx][(pixels_y.cpu(), pixels_x.cpu())].to(self.device)  # (B,3)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()
        # Unproject: direction in camera space
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3],
            p[:, :, None]
        ).squeeze()   # (B,3)
        # Rotate to world space: R^T @ p_cam
        R = self.pose_all[img_idx, :3, :3]  # (3,3)
        rays_v = torch.matmul(R.T[None], p[:, :, None]).squeeze()  # (B,3)
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)
        # Camera origin: R^T @ (-t)
        t = self.pose_all[img_idx, :3, 3]
        rays_o = (R.T @ (-t)).expand(rays_v.shape)  # (B,3)
        return torch.cat([rays_o, rays_v, color, mask[:, 0:1]], dim=-1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """Interpolate pose between two cameras (same OpenCV w2c convention)."""
        l  = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l, device=self.device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=self.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        p = torch.matmul(
            self.intrinsics_all_inv[0, None, None, :3, :3],
            p[:, :, :, None]
        ).squeeze()  # (W,H,3)  -- camera-space direction

        # w2c for both cameras
        w2c_0 = self.pose_all[idx_0].detach().cpu().numpy()
        w2c_1 = self.pose_all[idx_1].detach().cpu().numpy()
        # Convert to c2w for slerp
        c2w_0 = np.linalg.inv(w2c_0)
        c2w_1 = np.linalg.inv(w2c_1)

        rot_0 = c2w_0[:3, :3]
        rot_1 = c2w_1[:3, :3]
        rots  = Rot.from_matrix(np.stack([rot_0, rot_1]))
        slerp = Slerp([0, 1], rots)
        rot   = slerp(ratio)
        pose  = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3]  = ((1.0 - ratio) * c2w_0 + ratio * c2w_1)[:3, 3]
        # Convert interpolated c2w back to w2c
        w2c_interp = np.linalg.inv(pose)
        R_t   = torch.from_numpy(w2c_interp[:3, :3]).to(self.device)
        t_t   = torch.from_numpy(w2c_interp[:3, 3]).to(self.device)
        rays_v = torch.matmul(R_t.T[None, None], p[:, :, :, None]).squeeze()
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)
        rays_o = (R_t.T @ (-t_t)).expand(rays_v.shape)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        """Estimate near/far along rays assuming scene in unit sphere."""
        a   = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b   = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far  = mid + 1.0
        return near, far


    def image_at(self, idx, resolution_level):
        """Return a downsampled image (uint8 BGR) for visualisation."""
        # Re-read from disk to keep consistent with existing callers that expect BGR
        img_path = None
        # We don't store paths; reconstruct from images tensor
        img_np = (self.images[idx].numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,3) RGB
        img_bgr = img_np[:, :, ::-1]   # → BGR for cv.resize
        return cv.resize(
            img_bgr,
            (self.W // resolution_level, self.H // resolution_level)
        ).clip(0, 255)
