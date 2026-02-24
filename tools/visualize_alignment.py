#!/usr/bin/env python3
"""Visualize mesh-camera alignment by projecting mesh onto GT images.

Projects mesh faces (semi-transparent red) and edges (darker red) onto
camera images to verify that mesh and camera poses are correctly aligned.

Usage:
    python tools/visualize_alignment.py \
        --data-dir ../Dataset/3DGRUT/lego_glass \
        --split train \
        --mesh-key mesh_outside \
        --output-dir tools/alignment \
        --max-frames 5
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh


# =============================================================================
# Constants
# =============================================================================

FACE_COLOR = (0, 0, 200)       # BGR: semi-transparent red for faces
EDGE_COLOR = (0, 0, 120)       # BGR: darker red for edges
FACE_ALPHA = 0.3
EDGE_THICKNESS = 1


# =============================================================================
# Coordinate Transform Helpers
# =============================================================================

def build_intrinsic_matrix(meta, img_w, img_h):
    """Build 3x3 camera intrinsic matrix from NeRF JSON metadata.

    Args:
        meta (dict): JSON metadata with 'camera_angle_x'.
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.

    Returns:
        K (np.ndarray): Intrinsic matrix (3, 3).
    """
    fx = fy = 0.5 * img_w / np.tan(0.5 * meta["camera_angle_x"])
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def compute_w2c_opencv(c2w_nerf):
    """Convert NeRF (OpenGL) c2w to OpenCV world-to-camera matrix.

    NeRF convention: [right, up, back]
    OpenCV convention: [right, down, front]
    Flip Y and Z axes in camera space.

    Args:
        c2w_nerf (np.ndarray): Camera-to-world matrix (4, 4) in NeRF/OpenGL convention.

    Returns:
        w2c_cv (np.ndarray): World-to-camera matrix (4, 4) in OpenCV convention.
    """
    w2c = np.linalg.inv(c2w_nerf)
    # Flip Y and Z in camera space: OpenGL [right, up, back] → OpenCV [right, down, front]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return flip @ w2c


# =============================================================================
# Projection & Rendering
# =============================================================================

def project_vertices(vertices, w2c, K):
    """Project 3D vertices to 2D image coordinates.

    Args:
        vertices (np.ndarray): World-space vertices (V, 3).
        w2c (np.ndarray): World-to-camera matrix (4, 4), OpenCV convention.
        K (np.ndarray): Intrinsic matrix (3, 3).

    Returns:
        pts_2d (np.ndarray): Projected 2D points (V, 2), float.
        depths (np.ndarray): Depth values in camera space (V,).
    """
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ vertices.T).T + t  # (V, 3)
    depths = pts_cam[:, 2]

    # Perspective projection
    pts_2d = (K @ pts_cam.T).T  # (V, 3)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]  # (V, 2)

    return pts_2d, depths


def get_visible_faces(vertices, faces, face_normals, w2c):
    """Determine which faces are facing the camera (backface culling).

    Args:
        vertices (np.ndarray): World-space vertices (V, 3).
        faces (np.ndarray): Face indices (F, 3).
        face_normals (np.ndarray): Face normals in world space (F, 3).
        w2c (np.ndarray): World-to-camera matrix (4, 4), OpenCV convention.

    Returns:
        visible_mask (np.ndarray): Boolean mask (F,), True for front-facing faces.
    """
    # Camera position in world space
    cam_pos = np.linalg.inv(w2c)[:3, 3]

    # Face centroids
    centroids = vertices[faces].mean(axis=1)  # (F, 3)

    # View direction: from face centroid to camera
    view_dirs = cam_pos - centroids  # (F, 3)
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)

    # Face is visible if normal dot view_dir > 0
    dots = np.sum(face_normals * view_dirs, axis=1)
    return dots > 0


def render_mesh_overlay(image, mesh, w2c, K):
    """Render mesh overlay on image: semi-transparent red faces + darker red edges.

    Args:
        image (np.ndarray): BGR image (H, W, 3), uint8.
        mesh (trimesh.Trimesh): Mesh object in world space.
        w2c (np.ndarray): World-to-camera matrix (4, 4), OpenCV convention.
        K (np.ndarray): Intrinsic matrix (3, 3).

    Returns:
        overlay (np.ndarray): BGR image with mesh overlay (H, W, 3), uint8.
    """
    H, W = image.shape[:2]
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces)
    face_normals = np.array(mesh.face_normals, dtype=np.float64)

    # Project all vertices
    pts_2d, depths = project_vertices(vertices, w2c, K)

    # Backface culling
    visible = get_visible_faces(vertices, faces, face_normals, w2c)

    # Filter faces behind camera (all 3 vertices must have positive depth)
    face_depths = depths[faces]  # (F, 3)
    in_front = np.all(face_depths > 0, axis=1)  # (F,)
    visible = visible & in_front

    visible_faces = faces[visible]

    # Sort visible faces by mean depth (back-to-front for painter's algorithm)
    mean_depths = face_depths[visible].mean(axis=1)
    sort_order = np.argsort(-mean_depths)  # farthest first
    visible_faces = visible_faces[sort_order]

    # Draw faces: fill polygons on a copy, then alpha-blend with original
    face_overlay = image.copy()

    for face in visible_faces:
        tri_pts = pts_2d[face].astype(np.int32)  # (3, 2)
        # Bounds check: skip if all points outside image
        if np.all(tri_pts[:, 0] < 0) or np.all(tri_pts[:, 0] >= W):
            continue
        if np.all(tri_pts[:, 1] < 0) or np.all(tri_pts[:, 1] >= H):
            continue
        cv2.fillPoly(face_overlay, [tri_pts], FACE_COLOR)

    # Alpha blend: where face_overlay differs from image, the faces were drawn
    overlay = cv2.addWeighted(face_overlay, FACE_ALPHA, image, 1 - FACE_ALPHA, 0)

    # Draw edges (darker red) on visible faces
    drawn_edges = set()
    for face in visible_faces:
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            p1 = tuple(pts_2d[edge[0]].astype(int))
            p2 = tuple(pts_2d[edge[1]].astype(int))
            # Simple bounds check
            if (-W < p1[0] < 2 * W and -H < p1[1] < 2 * H and
                    -W < p2[0] < 2 * W and -H < p2[1] < 2 * H):
                cv2.line(overlay, p1, p2, EDGE_COLOR, EDGE_THICKNESS, cv2.LINE_AA)

    return overlay


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_info(data_dir, split, mesh_key):
    """Load camera poses, intrinsics, and mesh from a NeRF dataset.

    Args:
        data_dir (str): Dataset root directory.
        split (str): Dataset split ('train', 'val', 'test').
        mesh_key (str): Key in JSON for mesh path ('mesh_inside' or 'mesh_outside').

    Returns:
        frames (list[dict]): List of frame dicts from JSON.
        K (np.ndarray): Intrinsic matrix (3, 3).
        mesh (trimesh.Trimesh): Loaded mesh.
        img_wh (tuple[int, int]): Image (width, height).
    """
    json_path = os.path.join(data_dir, f"transforms_{split}.json")
    with open(json_path, "r") as f:
        meta = json.load(f)

    frames = meta["frames"]

    # Resolve image size
    first_path = os.path.join(data_dir, frames[0]["file_path"])
    for suffix in ["", ".png", ".jpg"]:
        if os.path.exists(first_path + suffix):
            img = cv2.imread(first_path + suffix)
            img_w, img_h = img.shape[1], img.shape[0]
            break
    else:
        img_w = int(meta.get("w", meta.get("width", 800)))
        img_h = int(meta.get("h", meta.get("height", 800)))

    K = build_intrinsic_matrix(meta, img_w, img_h)

    # Load mesh
    mesh_rel_path = meta.get(mesh_key)
    if mesh_rel_path is None:
        # Fallback: try reading from train JSON
        train_json = os.path.join(data_dir, "transforms_train.json")
        with open(train_json, "r") as f:
            train_meta = json.load(f)
        mesh_rel_path = train_meta.get(mesh_key)
    if mesh_rel_path is None:
        raise ValueError(f"Mesh key '{mesh_key}' not found in JSON")

    mesh_path = os.path.join(data_dir, mesh_rel_path)
    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded mesh: {mesh_path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    return frames, K, mesh, (img_w, img_h)


def load_image(data_dir, frame, img_wh):
    """Load a single image for a frame.

    Args:
        data_dir (str): Dataset root directory.
        frame (dict): Frame dict with 'file_path'.
        img_wh (tuple[int, int]): Expected (width, height).

    Returns:
        image (np.ndarray): BGR image (H, W, 3), uint8. None if not found.
    """
    base_path = os.path.join(data_dir, frame["file_path"])
    for suffix in ["", ".png", ".jpg"]:
        full_path = base_path + suffix
        if os.path.exists(full_path):
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Handle 16-bit images: convert to 8-bit
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)

            # Handle RGBA: composite onto black background
            if len(img.shape) == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                # Zero out RGB in transparent areas to avoid garbage color noise
                img = (rgb * alpha).astype(np.uint8)
            return img
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for mesh-camera alignment visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize mesh-camera alignment by projecting mesh onto GT images."
    )
    parser.add_argument("--data-dir", required=True, type=str,
                        help="Path to dataset root (e.g. data/ours_synthetic/lego)")
    parser.add_argument("--split", default="train", type=str,
                        choices=["train", "val", "test"],
                        help="Dataset split to use (default: train)")
    parser.add_argument("--mesh-key", default="mesh_outside", type=str,
                        help="Key in JSON for mesh path (default: mesh_outside)")
    parser.add_argument("--output-dir", default="output/alignment", type=str,
                        help="Output directory for overlay images")
    parser.add_argument("--max-frames", default=5, type=int,
                        help="Maximum number of frames to process (default: 5, -1 for all)")
    args = parser.parse_args()

    # Load data
    frames, K, mesh, img_wh = load_dataset_info(args.data_dir, args.split, args.mesh_key)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Limit frames
    n_frames = len(frames) if args.max_frames < 0 else min(args.max_frames, len(frames))
    print(f"Processing {n_frames}/{len(frames)} frames...")

    for i in range(n_frames):
        frame = frames[i]

        # Load GT image
        image = load_image(args.data_dir, frame, img_wh)
        if image is None:
            print(f"  [{i}] Skipping: image not found for {frame['file_path']}")
            continue

        # Get c2w in NeRF (OpenGL) convention — NO flip
        c2w_nerf = np.array(frame["transform_matrix"], dtype=np.float64)

        # Convert to OpenCV w2c
        w2c = compute_w2c_opencv(c2w_nerf)

        # Render overlay
        overlay = render_mesh_overlay(image, mesh, w2c, K)

        # Save
        frame_name = Path(frame["file_path"]).stem
        out_path = os.path.join(args.output_dir, f"align_{frame_name}.png")
        cv2.imwrite(out_path, overlay)
        print(f"  [{i}] Saved: {out_path}")

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
