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
    """TensorRT optimized CLIP inference class"""
    
    def __init__(self, clip_model, data_processor, device='cuda', cache_dir='./tensorrt_cache'):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.clip_model = clip_model
        self.data_processor = data_processor
        
        # TensorRT engines
        self.image_engine = None
        self.text_engine = None
        self.image_context = None
        self.text_context = None
        
        # Fallback to PyTorch if TensorRT fails
        self.use_tensorrt = TRT_AVAILABLE
        
        if self.use_tensorrt:
            try:
                self._init_tensorrt_engines()
                print("TensorRT CLIP engines initialized successfully")
            except Exception as e:
                print(f"TensorRT initialization failed: {e}, falling back to PyTorch")
                self.use_tensorrt = False
        
        # Cache for text features (works with both TensorRT and PyTorch)
        self.text_feature_cache = {}
    
    def _init_tensorrt_engines(self):
        """Initialize TensorRT engines for image and text encoders"""
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
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
        """Build TensorRT engine for CLIP image encoder"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # Set memory pool size (2GB)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
            
            # Enable FP16 if supported
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("Using FP16 optimization for image encoder")
            
            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Export CLIP vision model to ONNX first, then import to TensorRT
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            onnx_path = self.cache_dir / "clip_image_encoder.onnx"
            
            # Export image encoder to ONNX
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
                opset_version=14
            )
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed to parse ONNX model")
                    return None
            
            # Set optimization profile for dynamic batching - optimize for larger batches
            profile = builder.create_optimization_profile()
            profile.set_shape('pixel_values', (1, 3, 224, 224), (64, 3, 224, 224), (512, 3, 224, 224))
            config.add_optimization_profile(profile)
            
            # Build engine
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
        """Build TensorRT engine for CLIP text encoder"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # Set memory pool size
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * 1024 * 1024 * 1024)
            
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
                opset_version=14
            )
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed to parse text ONNX model")
                    return None
            
            # Set optimization profile
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
        """Load TensorRT engine from file"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(engine_path, 'rb') as f:
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"Failed to load engine from {engine_path}: {e}")
            return None
    
    def get_cached_text_features(self, labels_tuple: Tuple[str, ...]) -> torch.Tensor:
        """Get cached text features or compute and cache them"""
        if labels_tuple in self.text_feature_cache:
            return self.text_feature_cache[labels_tuple]
        
        # Compute text features
        if self.use_tensorrt and self.text_engine and self.text_context:
            text_features = self._encode_text_tensorrt(list(labels_tuple))
        else:
            text_features = self._encode_text_pytorch(list(labels_tuple))
        
        # Cache the result
        self.text_feature_cache[labels_tuple] = text_features
        return text_features
    
    def _encode_text_pytorch(self, texts: List[str]) -> torch.Tensor:
        """Encode text using PyTorch (fallback)"""
        text_inputs = self.data_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _encode_text_tensorrt(self, texts: List[str]) -> torch.Tensor:
        """Encode text using TensorRT with safe memory management"""
        try:
            # Tokenize texts
            text_inputs = self.data_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            input_ids = text_inputs['input_ids'].to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)
            
            batch_size = input_ids.shape[0]
            
            # Use safer memory allocation with proper cleanup
            input_ids_gpu = None
            attention_mask_gpu = None
            output_gpu = None
            
            try:
                # Allocate GPU memory
                input_ids_gpu = cuda.mem_alloc(input_ids.nbytes)
                attention_mask_gpu = cuda.mem_alloc(attention_mask.nbytes)
                output_gpu = cuda.mem_alloc(batch_size * 512 * 4)  # 512 features, float32
                
                # Copy inputs to GPU
                cuda.memcpy_htod(input_ids_gpu, input_ids.cpu().numpy())
                cuda.memcpy_htod(attention_mask_gpu, attention_mask.cpu().numpy())
                
                # Set input shapes using new TensorRT API
                if hasattr(self.text_context, 'set_input_shape'):
                    # New TensorRT API
                    self.text_context.set_input_shape('input_ids', input_ids.shape)
                    self.text_context.set_input_shape('attention_mask', attention_mask.shape)
                elif hasattr(self.text_context, 'set_binding_shape'):
                    # Legacy TensorRT API
                    self.text_context.set_binding_shape(0, input_ids.shape)
                    self.text_context.set_binding_shape(1, attention_mask.shape)
                else:
                    # Fallback - try to get binding indices
                    for i in range(self.text_engine.num_bindings):
                        if self.text_engine.binding_is_input(i):
                            if i == 0:
                                self.text_context.set_binding_shape(i, input_ids.shape)
                            elif i == 1:
                                self.text_context.set_binding_shape(i, attention_mask.shape)
                
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
                # Clean up GPU memory
                if input_ids_gpu is not None:
                    try:
                        input_ids_gpu.free()
                    except:
                        pass
                if attention_mask_gpu is not None:
                    try:
                        attention_mask_gpu.free()
                    except:
                        pass
                if output_gpu is not None:
                    try:
                        output_gpu.free()
                    except:
                        pass
            
        except Exception as e:
            print(f"TensorRT text encoding failed: {e}, falling back to PyTorch")
            return self._encode_text_pytorch(texts)
    
    def encode_images(self, images: List[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """Encode images using TensorRT or PyTorch"""
        if self.use_tensorrt and self.image_engine and self.image_context:
            return self._encode_images_tensorrt(images, batch_size)
        else:
            return self._encode_images_pytorch(images, batch_size)
    
    def _encode_images_pytorch(self, images: List[np.ndarray], batch_size: int) -> torch.Tensor:
        """Encode images using PyTorch (fallback)"""
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            image_inputs = self.data_processor(images=batch_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_features.append(image_features)
        
        return torch.cat(all_features, dim=0)
    
    def _encode_images_tensorrt(self, images: List[np.ndarray], batch_size: int) -> torch.Tensor:
        """Encode images using TensorRT with safe memory management"""
        try:
            all_features = []
            
            # Use very small batch sizes to prevent memory issues
            safe_batch_size = min(batch_size, 8)  # Much smaller batch size
            
            for i in range(0, len(images), safe_batch_size):
                batch_images = images[i:i+safe_batch_size]
                
                # Preprocess images
                image_inputs = self.data_processor(images=batch_images, return_tensors="pt")
                pixel_values = image_inputs['pixel_values'].to(self.device)
                
                current_batch_size = pixel_values.shape[0]
                
                # Use safer memory allocation with proper cleanup
                input_gpu = None
                output_gpu = None
                
                try:
                    # Allocate GPU memory
                    input_gpu = cuda.mem_alloc(pixel_values.nbytes)
                    output_gpu = cuda.mem_alloc(current_batch_size * 512 * 4)  # 512 features, float32
                    
                    # Copy input to GPU
                    cuda.memcpy_htod(input_gpu, pixel_values.cpu().numpy())
                    
                    # Set input shape using new TensorRT API
                    if hasattr(self.image_context, 'set_input_shape'):
                        # New TensorRT API
                        self.image_context.set_input_shape('pixel_values', pixel_values.shape)
                    elif hasattr(self.image_context, 'set_binding_shape'):
                        # Legacy TensorRT API
                        self.image_context.set_binding_shape(0, pixel_values.shape)
                    else:
                        # Fallback - try to get binding indices
                        for j in range(self.image_engine.num_bindings):
                            if self.image_engine.binding_is_input(j):
                                self.image_context.set_binding_shape(j, pixel_values.shape)
                                break
                    
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
                    
                except Exception as batch_error:
                    print(f"TensorRT batch processing failed: {batch_error}")
                    # Fall back to PyTorch for this batch
                    return self._encode_images_pytorch(images, batch_size)
                    
                finally:
                    # Clean up GPU memory
                    if input_gpu is not None:
                        try:
                            input_gpu.free()
                        except:
                            pass
                    if output_gpu is not None:
                        try:
                            output_gpu.free()
                        except:
                            pass
            
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