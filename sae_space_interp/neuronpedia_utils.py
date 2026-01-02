import json
import urllib.parse

def get_neuronpedia_quick_list_url(
    sae,
    feature_indices: list[int],
    list_name: str = "Quick List"
) -> str:
    """
    Builds a Neuronpedia quick list URL using information from the SAE config.
    
    Args:
        sae: The SAELens SAE object.
        feature_indices: List of SAE feature indices.
        list_name: Name for the quick list.
        
    Returns:
        The full Neuronpedia quick list URL.
    """
    # Prefer using neuronpedia_id from metadata if available
    neuronpedia_id = getattr(sae.cfg, "metadata", {}).get("neuronpedia_id")
    
    if neuronpedia_id:
        # Expected format: "model_id/layer_id"
        parts = neuronpedia_id.split("/")
        if len(parts) == 2:
            model_id, layer_id = parts
        else:
            # Fallback if format is unexpected
            model_id = sae.cfg.model_name
            layer_id = sae.cfg.hook_name
    else:
        # Generic fallback
        model_id = sae.cfg.model_name
        layer_id = sae.cfg.hook_name

    list_features = [
        {"modelId": model_id, "layer": layer_id, "index": str(idx)}
        for idx in feature_indices
    ]
    
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(list_name)
    url = url + "?name=" + name
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_features))
    
    return url
