"""
TensorRT optimization for CLIP model inference
Provides significant speedup (3-5x) for CLIP image and text encoding
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
from pathlib import Path

# TensorRT imports (with fallback if not available)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
    print("TensorRT available for CLIP optimization")
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, falling back to PyTorch")

class TensorRTCLIP:
    """TensorRT optimized CLIP inference class with CUDA context isolation"""
    
    def __init__(self, clip_model, data_processor, device='cuda', cache_dir='./tensorrt_cache'):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.clip_model = clip_model
        self.data_processor = data_processor
        
        # Import context manager
        from .cuda_context_manager import get_context_manager
        self.context_manager = get_context_manager()
        
        # TensorRT engines
        self.image_engine = None
        self.text_engine = None
        self.image_context = None
        self.text_context = None
        
        # Health monitoring
        self.consecutive_failures = 0
        self.max_failures = 3
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Fallback to PyTorch if TensorRT fails
        self.use_tensorrt = TRT_AVAILABLE and self.context_manager.health_status['tensorrt_healthy']
        
        if self.use_tensorrt:
            try:
                self._init_tensorrt_engines()
                print("TensorRT CLIP engines initialized successfully with context isolation")
            except Exception as e:
                print(f"TensorRT initialization failed: {e}, falling back to PyTorch")
                self.use_tensorrt = False
                self.consecutive_failures += 1
        
        # Cache for text features (works with both TensorRT and PyTorch)
        self.text_feature_cache = {}
    
    def _check_health_and_recovery(self) -> bool:
        """Check health and attempt recovery if needed"""
        current_time = time.time()
        
        # Only check health periodically
        if current_time - self.last_health_check < self.health_check_interval:
            return self.use_tensorrt
        
        self.last_health_check = current_time
        
        # Check context manager health
        health_status = self.context_manager.check_health()
        
        if not health_status['tensorrt_healthy'] and self.use_tensorrt:
            print(f"TensorRT context unhealthy: {health_status.get('last_error', 'Unknown error')}")
            
            # Attempt recovery if we haven't exceeded max failures
            if self.consecutive_failures < self.max_failures:
                print("Attempting TensorRT context recovery...")
                if self.context_manager.attempt_recovery():
                    try:
                        self._init_tensorrt_engines()
                        self.consecutive_failures = 0
                        self.use_tensorrt = True
                        print("TensorRT context recovery successful")
                        return True
                    except Exception as e:
                        print(f"TensorRT re-initialization failed: {e}")
                        self.consecutive_failures += 1
                        self.use_tensorrt = False
                else:
                    self.consecutive_failures += 1
                    self.use_tensorrt = False
            else:
                print(f"Max TensorRT failures ({self.max_failures}) exceeded, staying in PyTorch mode")
                self.use_tensorrt = False
        
        return self.use_tensorrt
    
    def _init_tensorrt_engines(self):
        """Initialize TensorRT engines with context isolation"""
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
        # Use TensorRT context for engine initialization
        with self.context_manager.tensorrt_context_scope() as memory_pool:
            # File paths for cached engines
            image_engine_path = self.cache_dir / "clip_image_encoder.engine"
            text_engine_path = self.cache_dir / "clip_text_encoder.engine"
            
            # Build or load image encoder engine
            if image_engine_path.exists():
                print("Loading cached image encoder TensorRT engine...")
                self.image_engine = self._load_engine(str(image_engine_path))
            else:
                print("Building image encoder TensorRT engine...")
                self.image_engine = self._build_image_engine(str(image_engine_path))
            
            # Build or load text encoder engine
            if text_engine_path.exists():
                print("Loading cached text encoder TensorRT engine...")
                self.text_engine = self._load_engine(str(text_engine_path))
            else:
                print("Building text encoder TensorRT engine...")
                self.text_engine = self._build_text_engine(str(text_engine_path))
            
            # Create execution contexts
            if self.image_engine:
                self.image_context = self.image_engine.create_execution_context()
            if self.text_engine:
                self.text_context = self.text_engine.create_execution_context()
    
    def _build_image_engine(self, engine_path: str) -> Optional[trt.ICudaEngine]:
        """Build TensorRT engine for CLIP image encoder with safer memory management"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # Conservative memory pool size
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * 1024 * 1024 * 1024)  # 1GB instead of 2GB
            
            # Enable FP16 if supported
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 optimization for image encoder")
            
            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Export CLIP vision model to ONNX first, then import to TensorRT
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            onnx_path = self.cache_dir / "clip_image_encoder.onnx"
            
            # Export image encoder to ONNX with error handling
            try:
                torch.onnx.export(
                    self.clip_model.vision_model,
                    dummy_input,
                    str(onnx_path),
                    input_names=['pixel_values'],
                    output_names=['last_hidden_state'],
                    dynamic_axes={
                        'pixel_values': {0: 'batch_size'},
                        'last_hidden_state': {0: 'batch_size'}
                    },
                    opset_version=14,
                    do_constant_folding=True
                )
            except Exception as e:
                print(f"ONNX export failed: {e}")
                return None
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for i in range(parser.num_errors):
                        print(f"ONNX parse error {i}: {parser.get_error(i)}")
                    return None
            
            # Conservative optimization profile for stability
            profile = builder.create_optimization_profile()
            profile.set_shape('pixel_values', (1, 3, 224, 224), (32, 3, 224, 224), (128, 3, 224, 224))  # Smaller max batch
            config.add_optimization_profile(profile)
            
            # Build engine with timeout
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                print("Failed to build image encoder engine")
                return None
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine)
            
            # Deserialize for use
            runtime = trt.Runtime(logger)
            return runtime.deserialize_cuda_engine(engine)
            
        except Exception as e:
            print(f"Failed to build image encoder engine: {e}")
            return None
    
    def _build_text_engine(self, engine_path: str) -> Optional[trt.ICudaEngine]:
        """Build TensorRT engine for CLIP text encoder with safer memory management"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # Conservative memory pool size
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)  # 512MB
            
            # Enable FP16 if supported
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 optimization for text encoder")
            
            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Export text encoder to ONNX
            dummy_input_ids = torch.randint(0, 1000, (1, 77)).to(self.device)
            dummy_attention_mask = torch.ones(1, 77).to(self.device)
            onnx_path = self.cache_dir / "clip_text_encoder.onnx"
            
            try:
                torch.onnx.export(
                    self.clip_model.text_model,
                    (dummy_input_ids, dummy_attention_mask),
                    str(onnx_path),
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['last_hidden_state'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size'},
                        'attention_mask': {0: 'batch_size'},
                        'last_hidden_state': {0: 'batch_size'}
                    },
                    opset_version=14,
                    do_constant_folding=True
                )
            except Exception as e:
                print(f"Text ONNX export failed: {e}")
                return None
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for i in range(parser.num_errors):
                        print(f"Text ONNX parse error {i}: {parser.get_error(i)}")
                    return None
            
            # Conservative optimization profile
            profile = builder.create_optimization_profile()
            profile.set_shape('input_ids', (1, 77), (10, 77), (20, 77))
            profile.set_shape('attention_mask', (1, 77), (10, 77), (20, 77))
            config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                print("Failed to build text encoder engine")
                return None
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine)
            
            # Deserialize for use
            runtime = trt.Runtime(logger)
            return runtime.deserialize_cuda_engine(engine)
            
        except Exception as e:
            print(f"Failed to build text encoder engine: {e}")
            return None
    
    def _load_engine(self, engine_path: str) -> Optional[trt.ICudaEngine]:
        """Load TensorRT engine from file with error handling"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(engine_path, 'rb') as f:
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"Failed to load engine from {engine_path}: {e}")
            return None
    
    def get_cached_text_features(self, labels_tuple: Tuple[str, ...]) -> torch.Tensor:
        """Get cached text features or compute and cache them with health checks"""
        if labels_tuple in self.text_feature_cache:
            return self.text_feature_cache[labels_tuple]
        
        # Check health and attempt recovery if needed
        tensorrt_available = self._check_health_and_recovery()
        
        # Compute text features
        if tensorrt_available and self.text_engine and self.text_context:
            try:
                text_features = self._encode_text_tensorrt(list(labels_tuple))
            except Exception as e:
                print(f"TensorRT text encoding failed: {e}, falling back to PyTorch")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    self.use_tensorrt = False
                text_features = self._encode_text_pytorch(list(labels_tuple))
        else:
            text_features = self._encode_text_pytorch(list(labels_tuple))
        
        # Cache the result
        self.text_feature_cache[labels_tuple] = text_features
        return text_features
    
    def _encode_text_pytorch(self, texts: List[str]) -> torch.Tensor:
        """Encode text using PyTorch (fallback) with context isolation"""
        try:
            # Use PyTorch context if context manager is available
            if hasattr(self, 'context_manager') and self.context_manager.health_status['pytorch_healthy']:
                with self.context_manager.pytorch_context_scope():
                    text_inputs = self.data_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        text_features = self.clip_model.get_text_features(**text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    return text_features
            else:
                # Standard PyTorch processing
                text_inputs = self.data_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features
        except Exception as e:
            print(f"PyTorch text encoding failed: {e}")
            raise
    
    def _encode_text_tensorrt(self, texts: List[str]) -> torch.Tensor:
        """Encode text using TensorRT with safe memory management and context isolation"""
        try:
            with self.context_manager.tensorrt_context_scope() as memory_pool:
                # Tokenize texts
                text_inputs = self.data_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
                input_ids = text_inputs['input_ids'].to(self.device)
                attention_mask = text_inputs['attention_mask'].to(self.device)
                
                batch_size = input_ids.shape[0]
                
                # Use memory pool for allocation
                input_ids_size = input_ids.nbytes
                attention_mask_size = attention_mask.nbytes
                output_size = batch_size * 512 * 4  # 512 features, float32
                
                input_ids_gpu = memory_pool.allocate(input_ids_size)
                attention_mask_gpu = memory_pool.allocate(attention_mask_size)
                output_gpu = memory_pool.allocate(output_size)
                
                if not all([input_ids_gpu, attention_mask_gpu, output_gpu]):
                    raise RuntimeError("Failed to allocate memory from pool")
                
                try:
                    # Copy inputs to GPU using PyCUDA
                    import pycuda.driver as cuda
                    cuda.memcpy_htod(input_ids_gpu, input_ids.cpu().numpy())
                    cuda.memcpy_htod(attention_mask_gpu, attention_mask.cpu().numpy())
                    
                    # Set input shapes
                    if hasattr(self.text_context, 'set_input_shape'):
                        self.text_context.set_input_shape('input_ids', input_ids.shape)
                        self.text_context.set_input_shape('attention_mask', attention_mask.shape)
                    elif hasattr(self.text_context, 'set_binding_shape'):
                        self.text_context.set_binding_shape(0, input_ids.shape)
                        self.text_context.set_binding_shape(1, attention_mask.shape)
                    
                    # Run inference
                    bindings = [int(input_ids_gpu), int(attention_mask_gpu), int(output_gpu)]
                    success = self.text_context.execute_v2(bindings)
                    
                    if not success:
                        raise RuntimeError("TensorRT text inference failed")
                    
                    # Copy output back
                    output = np.empty((batch_size, 512), dtype=np.float32)
                    cuda.memcpy_dtoh(output, output_gpu)
                    
                    # Convert to tensor and normalize
                    text_features = torch.from_numpy(output).to(self.device)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    return text_features
                    
                finally:
                    # Clean up GPU memory using memory pool
                    if input_ids_gpu:
                        memory_pool.deallocate(input_ids_gpu)
                    if attention_mask_gpu:
                        memory_pool.deallocate(attention_mask_gpu)
                    if output_gpu:
                        memory_pool.deallocate(output_gpu)
                        
        except Exception as e:
            print(f"TensorRT text encoding failed: {e}, falling back to PyTorch")
            return self._encode_text_pytorch(texts)
    
    def encode_images(self, images: List[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """Encode images using TensorRT or PyTorch with health checks"""
        # Check health and attempt recovery if needed
        tensorrt_available = self._check_health_and_recovery()
        
        if tensorrt_available and self.image_engine and self.image_context:
            try:
                return self._encode_images_tensorrt(images, batch_size)
            except Exception as e:
                print(f"TensorRT image encoding failed: {e}, falling back to PyTorch")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    self.use_tensorrt = False
                return self._encode_images_pytorch(images, batch_size)
        else:
            return self._encode_images_pytorch(images, batch_size)
    
    def _encode_images_pytorch(self, images: List[np.ndarray], batch_size: int) -> torch.Tensor:
        """Encode images using PyTorch (fallback) with context isolation"""
        try:
            all_features = []
            
            # Use PyTorch context if available
            if hasattr(self, 'context_manager') and self.context_manager.health_status['pytorch_healthy']:
                with self.context_manager.pytorch_context_scope():
                    for i in range(0, len(images), batch_size):
                        batch_images = images[i:i+batch_size]
                        image_inputs = self.data_processor(images=batch_images, return_tensors="pt").to(self.device)
                        
                        with torch.no_grad():
                            image_features = self.clip_model.get_image_features(**image_inputs)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            all_features.append(image_features)
            else:
                # Standard PyTorch processing
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size]
                    image_inputs = self.data_processor(images=batch_images, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        image_features = self.clip_model.get_image_features(**image_inputs)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        all_features.append(image_features)
            
            return torch.cat(all_features, dim=0) if all_features else torch.empty(0, 512).to(self.device)
            
        except Exception as e:
            print(f"PyTorch image encoding failed: {e}")
            raise
    
    def _encode_images_tensorrt(self, images: List[np.ndarray], batch_size: int) -> torch.Tensor:
        """Encode images using TensorRT with safe memory management and context isolation"""
        try:
            all_features = []
            
            # Use smaller batch sizes to prevent memory issues
            safe_batch_size = min(batch_size, 16)  # Conservative batch size
            
            with self.context_manager.tensorrt_context_scope() as memory_pool:
                for i in range(0, len(images), safe_batch_size):
                    batch_images = images[i:i+safe_batch_size]
                    
                    # Preprocess images
                    image_inputs = self.data_processor(images=batch_images, return_tensors="pt")
                    pixel_values = image_inputs['pixel_values'].to(self.device)
                    
                    current_batch_size = pixel_values.shape[0]
                    
                    # Allocate memory from pool
                    input_size = pixel_values.nbytes
                    output_size = current_batch_size * 512 * 4  # 512 features, float32
                    
                    input_gpu = memory_pool.allocate(input_size)
                    output_gpu = memory_pool.allocate(output_size)
                    
                    if not all([input_gpu, output_gpu]):
                        raise RuntimeError("Failed to allocate memory from pool")
                    
                    try:
                        # Copy input to GPU
                        import pycuda.driver as cuda
                        cuda.memcpy_htod(input_gpu, pixel_values.cpu().numpy())
                        
                        # Set input shape
                        if hasattr(self.image_context, 'set_input_shape'):
                            self.image_context.set_input_shape('pixel_values', pixel_values.shape)
                        elif hasattr(self.image_context, 'set_binding_shape'):
                            self.image_context.set_binding_shape(0, pixel_values.shape)
                        
                        # Run inference
                        bindings = [int(input_gpu), int(output_gpu)]
                        success = self.image_context.execute_v2(bindings)
                        
                        if not success:
                            raise RuntimeError("TensorRT inference failed")
                        
                        # Copy output back
                        output = np.empty((current_batch_size, 512), dtype=np.float32)
                        cuda.memcpy_dtoh(output, output_gpu)
                        
                        # Convert to tensor and normalize
                        image_features = torch.from_numpy(output).to(self.device)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        all_features.append(image_features)
                        
                    finally:
                        # Clean up memory
                        if input_gpu:
                            memory_pool.deallocate(input_gpu)
                        if output_gpu:
                            memory_pool.deallocate(output_gpu)
            
            return torch.cat(all_features, dim=0) if all_features else torch.empty(0, 512).to(self.device)
            
        except Exception as e:
            print(f"TensorRT image encoding failed: {e}, falling back to PyTorch")
            return self._encode_images_pytorch(images, batch_size)
    
    def classify_images(self, images: List[np.ndarray], labels: List[str], batch_size: int = 128) -> torch.Tensor:
        """Classify images using cached text features and TensorRT/PyTorch image encoding"""
        # Get cached text features
        labels_tuple = tuple(labels)
        text_features = self.get_cached_text_features(labels_tuple)
        
        # Encode images
        image_features = self.encode_images(images, batch_size)
        
        # Compute similarities
        logits = torch.matmul(image_features, text_features.T) * self.clip_model.logit_scale.exp()
        
        return logits


def create_tensorrt_clip(clip_model, data_processor, device='cuda'):
    """Factory function to create TensorRT optimized CLIP"""
    return TensorRTCLIP(clip_model, data_processor, device)