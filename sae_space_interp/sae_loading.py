# Reworked rom https://github.com/safety-research/open-source-em-features/

"""Utilities for loading Sparse Autoencoders (SAEs)."""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Any, Union, Optional
import os

try:
    from sae_lens import SAE
except ImportError:
    SAE = None


def load_sae_from_path(
    sae_path: Optional[str] = None,
    release: Optional[str] = None,
    sae_id: Optional[str] = None,
    device: str = "cpu"
) -> Any:
    """Load SAE using SAELens from either local path or pretrained release/id.
    
    Args:
        sae_path: Local directory path (e.g., "/path/to/sae")
        release: Release name for pretrained SAE (e.g., "llama_scope_lxr_32x")
        sae_id: SAE ID for pretrained SAE (e.g., "l2r_32x")
        device: Device to load the SAE on
        
    Returns:
        Loaded SAE model object from SAELens
        
    Note:
        Either provide (release, sae_id) for pretrained SAEs, or sae_path for local paths.
    """
    if SAE is None:
        raise ImportError(
            "sae_lens library is required to load SAEs. "
            "Please install it with: pip install sae-lens"
        )
    
    # Load from pretrained release/id
    if release is not None and sae_id is not None:
        print(f"Loading SAE from pretrained: release={release}, sae_id={sae_id}")
        try:
            sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device,
            )[0]
            sae.eval()
            
            print(f"SAE loaded on device: {device}")
            print(f"   Decoder shape: {sae.W_dec.shape}")
            
            return sae
        except Exception as e:
            raise RuntimeError(f"Failed to load SAE from pretrained (release={release}, sae_id={sae_id}): {e}")
    
    # Load from local path
    if sae_path is not None:
        print(f"Loading SAE from local path: {sae_path}")
        local_path = Path(sae_path)
        if not local_path.exists() or not local_path.is_dir():
            raise ValueError(f"SAE path does not exist or is not a directory: {sae_path}")
        
        try:
            sae = SAE.load_from_disk(
                str(local_path),
                device=device
            )
            sae.eval()
            
            print(f"SAE loaded on device: {device}")
            print(f"   Decoder shape: {sae.W_dec.shape}")
            
            return sae
        except Exception as e:
            raise RuntimeError(f"Failed to load SAE from {sae_path}: {e}")
    
    raise ValueError("Must provide either (release, sae_id) or sae_path")


