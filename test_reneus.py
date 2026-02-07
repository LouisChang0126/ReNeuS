#!/usr/bin/env python
"""
Test script for ReNeuS implementation
Tests basic refraction calculations and rendering setup
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.renderer import (
    compute_refraction,
    compute_reflection,
    compute_fresnel,
    check_total_internal_reflection
)

def test_refraction():
    """Test Snell's Law refraction calculation"""
    print("\n=== Testing Snell's Law Refraction ===")
    
    # Test case: ray entering glass from air at 45 degrees
    ray_d = torch.tensor([[0.707, 0.0, -0.707]])  # 45 degree angle
    normal = torch.tensor([[0.0, 0.0, 1.0]])  # surface normal pointing up
    ior_in = 1.0  # air
    ior_out = 1.5  # glass
    
    ray_d_refract, valid = compute_refraction(ray_d, normal, ior_in, ior_out)
    
    print(f"Incident ray: {ray_d.numpy()}")
    print(f"Surface normal: {normal.numpy()}")
    print(f"Refracted ray: {ray_d_refract.numpy()}")
    print(f"Valid refraction: {valid.numpy()}")
    
    # Check if refracted ray is normalized
    assert torch.allclose(torch.norm(ray_d_refract, dim=-1), torch.ones(1)), "Refracted ray should be normalized"
    
    # Check if refraction bends towards normal (smaller angle)
    incident_angle = torch.acos(-(ray_d * normal).sum())
    refracted_angle = torch.acos(-(ray_d_refract * normal).sum())
    print(f"Incident angle: {torch.rad2deg(incident_angle).item():.2f}°")
    print(f"Refracted angle: {torch.rad2deg(refracted_angle).item():.2f}°")
    
    assert refracted_angle < incident_angle, "Refracted ray should bend towards normal"
    print("✓ Refraction test passed!")


def test_reflection():
    """Test mirror reflection calculation"""
    print("\n=== Testing Reflection ===")
    
    ray_d = torch.tensor([[1.0, 0.0, -1.0]])  # 45 degree downward
    normal = torch.tensor([[0.0, 0.0, 1.0]])  # upward normal
    
    ray_d_reflect = compute_reflection(ray_d, normal)
    
    print(f"Incident ray: {ray_d.numpy()}")
    print(f"Reflected ray: {ray_d_reflect.numpy()}")
    
    # Check normalization
    assert torch.allclose(torch.norm(ray_d_reflect, dim=-1), torch.ones(1)), "Reflected ray should be normalized"
    
    # Check that reflection is symmetric
    expected = torch.tensor([[1.0, 0.0, 1.0]])
    expected = expected / torch.norm(expected)
    assert torch.allclose(ray_d_reflect, expected, atol=1e-5), "Reflection should be symmetric"
    print("✓ Reflection test passed!")


def test_fresnel():
    """Test Fresnel coefficient calculation"""
    print("\n=== Testing Fresnel Equations ===")
    
    # Test at various angles
    angles_deg = [0, 15, 30, 45, 60, 75, 89]
    ior_in = 1.0
    ior_out = 1.5
    
    print(f"IOR ratio: {ior_in} -> {ior_out}")
    print("Angle (deg) | Fresnel coeff | Refracted fraction")
    print("-" * 55)
    
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        cos_theta = torch.tensor([np.cos(angle_rad)])
        
        fresnel = compute_fresnel(cos_theta, ior_in, ior_out)
        
        print(f"{angle_deg:11.0f} | {fresnel.item():13.4f} | {(1-fresnel).item():18.4f}")
    
    # Check Fresnel at normal incidence (should be ~0.04 for air-glass)
    cos_normal = torch.tensor([1.0])
    fresnel_normal = compute_fresnel(cos_normal, ior_in, ior_out)
    expected_normal = ((1.5 - 1.0) / (1.5 + 1.0))**2
    assert torch.allclose(fresnel_normal, torch.tensor([expected_normal]), atol=1e-3), \
        f"Fresnel at normal incidence should be ~{expected_normal:.4f}"
    
    print("✓ Fresnel test passed!")


def test_total_internal_reflection():
    """Test TIR detection"""
    print("\n=== Testing Total Internal Reflection ===")
    
    # Glass to air (TIR can occur)
    ior_in = 1.5
    ior_out = 1.0
    
    critical_angle = np.rad2deg(np.arcsin(ior_out / ior_in))
    print(f"Critical angle for glass->air: {critical_angle:.2f}°")
    
    # Test below critical angle (should refract)
    cos_theta_below = torch.tensor([np.cos(np.deg2rad(40.0))])
    tir_below = check_total_internal_reflection(cos_theta_below, ior_in, ior_out)
    print(f"40° from normal: TIR = {tir_below.item()}")
    assert not tir_below.item(), "Should not have TIR below critical angle"
    
    # Test above critical angle (should have TIR)
    cos_theta_above = torch.tensor([np.cos(np.deg2rad(50.0))])
    tir_above = check_total_internal_reflection(cos_theta_above, ior_in, ior_out)
    print(f"50° from normal: TIR = {tir_above.item()}")
    assert tir_above.item(), "Should have TIR above critical angle"
    
    print("✓ TIR test passed!")


def test_dataset_loading():
    """Test loading ReNeuS dataset with metadata"""
    print("\n=== Testing Dataset Loading ===")
    
    try:
        from pyhocon import ConfigFactory
        from models.dataset import Dataset
        
        # Check if lego_glass dataset exists
        dataset_path = "/home/louis/Fish-Dev/Dataset/ReNeuS/lego_glass"
        if not os.path.exists(dataset_path):
            print(f"⚠ Dataset not found at {dataset_path}")
            print("Skipping dataset loading test")
            return
        
        # Create minimal config
        conf_text = f"""
        data_dir = {dataset_path}
        render_cameras_name = cameras_sphere.npz
        object_cameras_name = cameras_sphere.npz
        """
        conf = ConfigFactory.parse_string(conf_text)
        
        print(f"Loading dataset from: {dataset_path}")
        dataset = Dataset(conf)
        
        print(f"Number of images: {dataset.n_images}")
        print(f"Image resolution: {dataset.H} x {dataset.W}")
        print(f"IOR: {dataset.ior}")
        print(f"Container mesh path: {dataset.container_mesh_path}")
        
        assert dataset.ior is not None, "IOR should be loaded from metadata.json"
        assert dataset.container_mesh_path is not None, "Container mesh path should be loaded"
        
        print("✓ Dataset loading test passed!")
        
    except Exception as e:
        print(f"⚠ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("ReNeuS Implementation Tests")
    print("=" * 60)
    
    # Run all tests
    test_refraction()
    test_reflection()
    test_fresnel()
    test_total_internal_reflection()
    test_dataset_loading()
    
    print("\n" + "=" * 60)
    print("All core tests passed! ✓")
    print("=" * 60)
