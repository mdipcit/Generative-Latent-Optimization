"""
Model configuration definitions for VAE and other models.
"""

from typing import Dict, Any, Optional
import os


# Default model configurations for Stable Diffusion VAE models
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sd14": {
        "repo_id": "CompVis/stable-diffusion-v1-4",
        "subfolder": "vae",
        "description": "Stable Diffusion v1.4 VAE",
        "input_size": (512, 512),
        "latent_channels": 4,
    },
    "sd15": {
        "repo_id": "runwayml/stable-diffusion-v1-5", 
        "subfolder": "vae",
        "description": "Stable Diffusion v1.5 VAE",
        "input_size": (512, 512),
        "latent_channels": 4,
    },
}

# Default Hugging Face token (can be overridden via environment variable)
DEFAULT_HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", "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")


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'sd14', 'sd15')
        
    Returns:
        Dictionary containing model configuration
        
    Raises:
        ValueError: If model_name is not found
    """
    if model_name not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
    return MODEL_CONFIGS[model_name].copy()


def get_all_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get all available model configurations.
    
    Returns:
        Dictionary of all model configurations
    """
    return MODEL_CONFIGS.copy()


def add_model_config(model_name: str, config: Dict[str, Any]) -> None:
    """
    Add a new model configuration.
    
    Args:
        model_name: Name of the model
        config: Model configuration dictionary
    """
    MODEL_CONFIGS[model_name] = config


def list_available_models() -> list[str]:
    """
    List all available model names.
    
    Returns:
        List of available model names
    """
    return list(MODEL_CONFIGS.keys())


def get_default_token() -> Optional[str]:
    """
    Get the default Hugging Face token.
    
    Returns:
        Default token or None
    """
    return DEFAULT_HF_TOKEN