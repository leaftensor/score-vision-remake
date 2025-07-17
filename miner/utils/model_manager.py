from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO
from loguru import logger
import os

from miner.utils.device import get_optimal_device
from scripts.download_models import download_models

class ModelManager:
    """Manages the loading and caching of YOLO models."""
    
    def __init__(self, device: Optional[str] = None):
        # get_optimal_device accepts None, so pass as is
        self.device = get_optimal_device(device)
        self.models: Dict[str, YOLO] = {}
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Define model paths
        self.model_paths = {
            "player": self.data_dir / "football-player-detection.pt",
            "pitch": self.data_dir / "football-pitch-detection.pt",
            "ball": self.data_dir / "football-ball-detection.pt"
        }
        # TensorRT engine paths (same name, .engine extension)
        self.trt_paths = {
            name: Path(str(path).replace('.pt', '.engine'))
            for name, path in self.model_paths.items()
        }
        # Check if models exist, download if missing
        self._ensure_models_exist()

    def _ensure_models_exist(self) -> None:
        """Check if required models exist, download if missing."""
        missing_models = [
            name for name, path in self.model_paths.items() 
            if not path.exists()
        ]
        if missing_models:
            logger.info(f"Missing models: {', '.join(missing_models)}")
            logger.info("Downloading required models...")
            download_models()

    def load_trt_model(self, engine_path: Path) -> YOLO:
        """Load a YOLO model from a TensorRT .engine file."""
        logger.info(f"Loading TensorRT model from {engine_path}")
        # Ultralytics YOLO supports loading .engine directly
        model = YOLO(str(engine_path), task='detect')
        return model

    def load_model(self, model_name: str) -> YOLO:
        """
        Load a model by name, using cache if available. Prefer TensorRT if available.
        Args:
            model_name: Name of the model to load ('player', 'pitch', or 'ball')
        Returns:
            YOLO: The loaded model
        """
        if model_name in self.models:
            return self.models[model_name]
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        # Prefer TensorRT engine if exists
        trt_path = self.trt_paths[model_name]
        if trt_path.exists():
            model = self.load_trt_model(trt_path)
            logger.info(f"Loaded {model_name} as TensorRT engine: {trt_path}")
        else:
            model_path = self.model_paths[model_name]
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    "Please ensure all required models are downloaded."
                )
            logger.info(f"Loading {model_name} model from {model_path} to {self.device}")
            model = YOLO(str(model_path), task='detect').to(device=self.device)
        self.models[model_name] = model
        return model

    def load_all_models(self) -> None:
        """Load all models into cache."""
        for model_name in self.model_paths.keys():
            self.load_model(model_name)
    
    def get_model(self, model_name: str) -> YOLO:
        """
        Get a model by name, loading it if necessary.
        Args:
            model_name: Name of the model to get ('player', 'pitch', or 'ball')
        Returns:
            YOLO: The requested model
        """
        return self.load_model(model_name)
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.models.clear() 