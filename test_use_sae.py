#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to verify use_sae parameter functionality
"""
import torch
from models.topofusagnet import TopoFuSAGNet, JointLoss

def test_use_sae_enabled():
    """Test TopoFuSAGNet with use_sae=1 (default)"""
    print("\n=== Testing use_sae=1 (SAE Enabled) ===")
    
    bsz, num_nodes, win = 8, 20, 15
    x = torch.randn(bsz, num_nodes, win)
    y = torch.randn(bsz, num_nodes)
    
    model = TopoFuSAGNet(
        num_nodes=num_nodes,
        window_size=win,
        c=16,
        z_dim=12,
        emb_dim=10,
        gat_out_dim=32,
        topk=5,
        rho=0.05,
        dropout=0.1,
        gat_heads=2,
        use_sae=1,  # SAE enabled
    )
    
    predicted_vals, reconstructed_vals, kl_sparsity, sparsity_dev = model(x)
    
    # Verify outputs
    assert predicted_vals.shape == (bsz, num_nodes), f"Predicted shape mismatch: {predicted_vals.shape}"
    assert reconstructed_vals is not None, "Reconstruction should not be None with use_sae=1"
    assert reconstructed_vals.shape == (bsz, num_nodes, win), f"Reconstruction shape mismatch: {reconstructed_vals.shape}"
    assert isinstance(kl_sparsity, torch.Tensor), "KL sparsity should be a tensor"
    assert kl_sparsity > 0, "KL sparsity should be > 0"
    assert sparsity_dev.shape == (bsz, num_nodes), "sparsity_dev shape mismatch"
    
    # Test loss
    criterion = JointLoss(lambda_forecast=0.7, beta=1e-3, use_sae=1)
    loss_dict = criterion(
        predicted_vals=predicted_vals,
        forecast_target=y,
        reconstructed_vals=reconstructed_vals,
        recon_target=x,
        kl_sparsity=kl_sparsity,
    )
    
    assert loss_dict["total"] > 0, "Total loss should be > 0"
    assert loss_dict["forecast"] > 0, "Forecast loss should be > 0"
    assert loss_dict["reconstruction"] > 0, "Reconstruction loss should be > 0"
    assert loss_dict["sparsity"] > 0, "Sparsity loss should be > 0"
    
    print(f"✓ use_sae=1 outputs verified")
    print(f"  - predicted_vals: {predicted_vals.shape}")
    print(f"  - reconstructed_vals: {reconstructed_vals.shape}")
    print(f"  - kl_sparsity: {kl_sparsity.item():.6f}")
    print(f"  - total loss: {loss_dict['total'].item():.6f}")
    print(f"  - forecast loss: {loss_dict['forecast'].item():.6f}")
    print(f"  - reconstruction loss: {loss_dict['reconstruction'].item():.6f}")
    print(f"  - sparsity loss: {loss_dict['sparsity'].item():.6f}")


def test_use_sae_disabled():
    """Test TopoFuSAGNet with use_sae=0 (Ablation)"""
    print("\n=== Testing use_sae=0 (SAE Disabled - Ablation) ===")
    
    bsz, num_nodes, win = 8, 20, 15
    x = torch.randn(bsz, num_nodes, win)
    y = torch.randn(bsz, num_nodes)
    
    model = TopoFuSAGNet(
        num_nodes=num_nodes,
        window_size=win,
        c=16,
        z_dim=12,
        emb_dim=10,
        gat_out_dim=32,
        topk=5,
        rho=0.05,
        dropout=0.1,
        gat_heads=2,
        use_sae=0,  # SAE disabled
    )
    
    predicted_vals, reconstructed_vals, kl_sparsity, sparsity_dev = model(x)
    
    # Verify outputs
    assert predicted_vals.shape == (bsz, num_nodes), f"Predicted shape mismatch: {predicted_vals.shape}"
    assert reconstructed_vals is None, "Reconstruction should be None with use_sae=0"
    assert isinstance(kl_sparsity, torch.Tensor), "KL sparsity should be a tensor"
    assert kl_sparsity.item() == 0.0, "KL sparsity should be 0.0 with use_sae=0"
    assert sparsity_dev.shape == (bsz, num_nodes), "sparsity_dev shape mismatch"
    assert torch.allclose(sparsity_dev, torch.zeros_like(sparsity_dev)), "sparsity_dev should be 0 with use_sae=0"
    
    # Test loss
    criterion = JointLoss(lambda_forecast=0.7, beta=1e-3, use_sae=0)
    loss_dict = criterion(
        predicted_vals=predicted_vals,
        forecast_target=y,
        reconstructed_vals=reconstructed_vals,
        recon_target=x,
        kl_sparsity=kl_sparsity,
    )
    
    assert loss_dict["total"] > 0, "Total loss should be > 0"
    assert loss_dict["forecast"] > 0, "Forecast loss should be > 0"
    assert loss_dict["reconstruction"].item() == 0.0, "Reconstruction loss should be 0.0 with use_sae=0"
    assert loss_dict["sparsity"].item() == 0.0, "Sparsity loss should be 0.0 with use_sae=0"
    assert torch.allclose(loss_dict["total"], loss_dict["forecast"]), "Total loss should equal forecast loss with use_sae=0"
    
    print(f"✓ use_sae=0 outputs verified")
    print(f"  - predicted_vals: {predicted_vals.shape}")
    print(f"  - reconstructed_vals: {reconstructed_vals}")
    print(f"  - kl_sparsity: {kl_sparsity.item():.6f}")
    print(f"  - total loss: {loss_dict['total'].item():.6f}")
    print(f"  - forecast loss: {loss_dict['forecast'].item():.6f}")
    print(f"  - reconstruction loss: {loss_dict['reconstruction'].item():.6f}")
    print(f"  - sparsity loss: {loss_dict['sparsity'].item():.6f}")


def test_checkpoint_compatibility():
    """Test checkpoint loading with strict=False"""
    print("\n=== Testing Checkpoint Compatibility ===")
    
    import tempfile
    import os
    
    bsz, num_nodes, win = 8, 20, 15
    x = torch.randn(bsz, num_nodes, win)
    
    # Train model with use_sae=1
    model_sae = TopoFuSAGNet(
        num_nodes=num_nodes,
        window_size=win,
        c=16,
        z_dim=12,
        emb_dim=10,
        gat_out_dim=32,
        topk=5,
        use_sae=1,
    )
    
    # Save checkpoint to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint_sae.pt")
        torch.save(model_sae.state_dict(), checkpoint_path)
        
        # Load into model with use_sae=0 (different architecture)
        model_no_sae = TopoFuSAGNet(
            num_nodes=num_nodes,
            window_size=win,
            c=16,
            z_dim=12,
            emb_dim=10,
            gat_out_dim=32,
            topk=5,
            use_sae=0,
        )
        
        # Load with strict=False (should ignore missing/extra keys)
        state_dict = torch.load(checkpoint_path)
        incompatible = model_no_sae.load_state_dict(state_dict, strict=False)
        
        print(f"✓ Checkpoint loading with strict=False works")
        print(f"  - Missing keys: {incompatible.missing_keys}")
        print(f"  - Unexpected keys: {incompatible.unexpected_keys}")
        
        # Verify model still works
        output = model_no_sae(x)
        assert output[0].shape == (bsz, num_nodes), "Model should still produce valid output"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing use_sae Parameter Implementation")
    print("=" * 60)
    
    test_use_sae_enabled()
    test_use_sae_disabled()
    test_checkpoint_compatibility()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
