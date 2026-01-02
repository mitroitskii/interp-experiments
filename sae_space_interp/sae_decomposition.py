# Reworked rom https://github.com/safety-research/open-source-em-features/

"""SAE decomposition module for analyzing difference vectors."""

from __future__ import annotations

import json
import torch
import einops
from pathlib import Path
from typing import Dict, Any, Optional, Union

import sae_loading
from sae_loading import load_sae_from_path


def _decompose_vector_on_sae(
    vector: torch.Tensor,
    sae: Any,
    normalize: bool = False,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """Decompose a vector onto the SAE basis.
    
    Args:
        vector: Vector to decompose (d_model,)
        sae: SAE model object from SAELens (has W_dec attribute)
        normalize: Whether to normalize both vector and SAE features before computing similarities
        top_k: If specified, return only top-k and bottom-k features
        
    Returns:
        Dict with decomposition results
    """
    # Get decoder weight from SAE (SAELens uses W_dec with shape (n_features, d_model))
    decoder = sae.W_dec.float()
    
    # Ensure vector is the right shape
    if vector.dim() != 1:
        raise ValueError(f"Expected 1D vector, got shape {vector.shape}")
    v = vector.float()
    
    # Move to same device
    v = v.to(decoder.device)
    
    # SAELens W_dec is shaped (n_features, d_model)
    n_features, d_model = decoder.shape
    if d_model != v.shape[0]:
        raise ValueError(
            f"Vector dimension {v.shape[0]} does not match SAE decoder d_model {d_model}"
        )
    
    if normalize:
        # Cosine similarity: normalize feature vectors and input vector
        v_norm = v / (v.norm() + 1e-8)
        feat_vecs = decoder  # (n_features, d_model)
        feat_vecs = feat_vecs / (feat_vecs.norm(dim=1, keepdim=True) + 1e-8)
        similarities = einops.einsum(feat_vecs, v_norm, "n_features d_model, d_model -> n_features")
    else:
        # Dot product similarity
        similarities = einops.einsum(decoder, v, "n_features d_model, d_model -> n_features")
    
    # Sort by similarity (descending)
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    
    result = {
        "similarities": similarities,
        "sorted_similarities": sorted_similarities, 
        "sorted_indices": sorted_indices
    }
    
    if top_k is not None:
        n_features = len(similarities)
        k = min(top_k, n_features // 2)  # Don't exceed half the features
        
        # Top-k features (highest similarity)
        top_indices = sorted_indices[:k]
        top_similarities = sorted_similarities[:k]
        
        # Bottom-k features (most negative similarity, ascending order)
        bottom_indices = sorted_indices[-k:].flip(0)  # Use tensor flip instead of list slicing
        bottom_similarities = sorted_similarities[-k:].flip(0)
        
        result["top_features"] = {
            "indices": top_indices.cpu().tolist(),
            "similarities": top_similarities.cpu().tolist(),
            "count": k
        }
        
        result["bottom_features"] = {
            "indices": bottom_indices.cpu().tolist(),
            "similarities": bottom_similarities.cpu().tolist(), 
            "count": k
        }
    
    return result


def decompose_difference_vectors_on_sae(
    vector: torch.Tensor,
    sae_path: Optional[str] = None,
    layer: int = 0,
    *,
    sae: Optional[Any] = None,
    release: Optional[str] = None,
    sae_id: Optional[str] = None,
    device: str = "cpu",
    top_k: int = 100,
    out_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Decompose a vector onto SAE basis for interpretability.
    
    This function takes a vector (e.g. difference vector between models) and
    projects it onto the Sparse Autoencoder (SAE) feature basis to identify
    which interpretable features are most relevant.
    
    Args:
        vector: Vector to decompose (d_model,)
        sae_path: Optional path to SAE (local directory). Use if loading from local path.
        layer: Layer index (for metadata and reporting)
        sae: Optional pre-loaded SAE object. If provided, skips loading.
        release: Optional release name for pretrained SAE (e.g., "llama_scope_lxr_32x")
        sae_id: Optional SAE ID for pretrained SAE (e.g., "l2r_32x")
        device: Device for computation
        top_k: Number of top/bottom features to return and save
        out_dir: Optional output directory to save results
        
    Returns:
        Dict with:
            - "layer": layer index analyzed
            - "sae_path": path to SAE used (or "pretrained" if using release/id)
            - "vector_norm": norm of the input vector
            - "top_features": list of top-k features with similarities
            - "bottom_features": list of bottom-k features with similarities
            - "decomposition_summary": summary statistics
    """
    if vector.dim() != 1:
        raise ValueError(f"Expected 1D vector, got shape {vector.shape}")
    
    print(f"Decomposing layer {layer} vector (norm: {vector.norm().item():.4f}) onto SAE...")
    
    # Load SAE if not provided
    if sae is None:
        if sae_path is not None:
            print(f"Loading SAE from: {sae_path}")
            sae = load_sae_from_path(sae_path=sae_path, device=device)
            sae_identifier = sae_path
        elif release is not None and sae_id is not None:
            print(f"Loading SAE from pretrained: release={release}, sae_id={sae_id}")
            sae = load_sae_from_path(release=release, sae_id=sae_id, device=device)
            sae_identifier = f"pretrained:{release}:{sae_id}"
        else:
            raise ValueError("Must provide either sae, sae_path, or (release, sae_id)")
    else:
        print(f"Using provided SAE object")
        sae_identifier = "provided_sae"
    
    n_features = sae.W_dec.shape[0]
    print(f"SAE loaded: {n_features} features")
    
    # Decompose the vector
    decomposition = _decompose_vector_on_sae(
        vector=vector.to(device),
        sae=sae,
        normalize=True,
        top_k=top_k
    )
    
    # Prepare results
    result = {
        "layer": layer,
        "sae_path": sae_identifier,
        "vector_norm": vector.norm().item(),
        "n_features": n_features,
        "top_features": decomposition["top_features"],
        "bottom_features": decomposition["bottom_features"],
        "decomposition_summary": {
            "max_similarity": decomposition["sorted_similarities"][0].item(),
            "min_similarity": decomposition["sorted_similarities"][-1].item(),
            "mean_abs_similarity": decomposition["similarities"].abs().mean().item(),
            "std_similarity": decomposition["similarities"].std().item(),
            "top_k": top_k,
            "sparsity": (decomposition["similarities"].abs() > 1e-6).float().mean().item()
        }
    }
    
    # Print summary
    print(f"Decomposition summary:")
    print(f"   Vector norm: {result['vector_norm']:.4f}")
    print(f"   Max similarity: {result['decomposition_summary']['max_similarity']:.4f}")
    print(f"   Min similarity: {result['decomposition_summary']['min_similarity']:.4f}")
    print(f"   Mean |similarity|: {result['decomposition_summary']['mean_abs_similarity']:.4f}")
    print(f"   Top {top_k} features: {result['top_features']['similarities'][:5]}...")
    print(f"   Bottom {top_k} features: {result['bottom_features']['similarities'][-5:]}...")
    
    # Save results if output directory specified
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = out_path / f"sae_decomposition_layer_{layer}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved SAE decomposition results to {results_file}")
        
        # Save detailed feature lists for easy analysis
        top_features_file = out_path / f"top_{top_k}_features_layer_{layer}.json"
        with open(top_features_file, 'w') as f:
            json.dump({
                "layer": layer,
                "sae_path": sae_identifier,
                "vector_norm": result["vector_norm"],
                "features": [
                    {
                        "rank": i + 1,
                        "feature_id": result["top_features"]["indices"][i],
                        "cosine_similarity": result["top_features"]["similarities"][i]
                    }
                    for i in range(len(result["top_features"]["indices"]))
                ]
            }, f, indent=2)
        
        bottom_features_file = out_path / f"bottom_{top_k}_features_layer_{layer}.json"
        with open(bottom_features_file, 'w') as f:
            json.dump({
                "layer": layer,
                "sae_path": sae_identifier,
                "vector_norm": result["vector_norm"],
                "features": [
                    {
                        "rank": i + 1,
                        "feature_id": result["bottom_features"]["indices"][i],
                        "cosine_similarity": result["bottom_features"]["similarities"][i]
                    }
                    for i in range(len(result["bottom_features"]["indices"]))
                ]
            }, f, indent=2)
        
        print(f"Saved detailed feature lists to {top_features_file} and {bottom_features_file}")
    
    return result


def decompose_all_layers_on_sae(
    difference_vectors: Dict[int, torch.Tensor],
    sae_path_template: str,
    *,
    device: str = "cpu", 
    top_k: int = 100,
    out_dir: Optional[Union[str, Path]] = None
) -> Dict[int, Dict[str, Any]]:
    """Decompose difference vectors for all layers using layer-specific SAEs.
    
    Args:
        difference_vectors: Dict mapping layer_idx to difference tensor
        sae_path_template: Template for SAE paths with {layer} placeholder
                          e.g., "andyrdt/saes-llama-3.1-8b-instruct/resid_post_layer_{layer}/trainer_1"
        device: Device for computation
        top_k: Number of top/bottom features to return
        out_dir: Output directory for results
        
    Returns:
        Dict mapping layer_idx to decomposition results
    """
    results = {}
    
    for layer_idx in sorted(difference_vectors.keys()):
        print(f"\n🔄 Processing layer {layer_idx}...")
        
        # Format SAE path for this layer
        sae_path = sae_path_template.format(layer=layer_idx)
        
        try:
            layer_result = decompose_difference_vectors_on_sae(
                vector=difference_vectors[layer_idx],
                sae_path=sae_path,
                layer=layer_idx,
                device=device,
                top_k=top_k,
                out_dir=out_dir
            )
            results[layer_idx] = layer_result
            
        except Exception as e:
            print(f"❌ Failed to decompose layer {layer_idx}: {e}")
            results[layer_idx] = {
                "error": str(e),
                "layer": layer_idx,
                "sae_path": sae_path
            }
    
    # Save combined results
    if out_dir is not None:
        out_path = Path(out_dir)
        combined_file = out_path / "sae_decomposition_all_layers.json"
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved combined results to {combined_file}")
    
    print(f"\n✅ Completed SAE decomposition for {len(results)} layers")
    return results
