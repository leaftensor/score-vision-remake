"""
CUDA Context Manager for TensorRT CLIP Stabilization

This module implements comprehensive CUDA context isolation and memory management
to fix the CUDA context corruption issues in TensorRT CLIP processing.

Addresses the architecture requirements for:
- CUDA Context Isolation
- Memory Pool Management
- Graceful Degradation
- Real-time Health Monitoring
- Context Recovery Mechanisms
"""

import threading
import time
import weakref
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable
import logging

import torch
import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None

logger = logging.getLogger(__name__)


class MemoryPool:
    """Pre-allocated CUDA memory pool for safe TensorRT operations"""
    
    def __init__(self, pool_size: int = 2 * 1024 * 1024 * 1024):  # 2GB default
        self.pool_size = pool_size
        self.allocated_blocks = {}
        self.free_blocks = []
        self.lock = threading.Lock()
        self.pool_ptr = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the memory pool"""
        if not PYCUDA_AVAILABLE:
            return False
            
        try:
            with self.lock:
                if self.initialized:
                    return True
                    
                # Allocate the main memory pool
                self.pool_ptr = cuda.mem_alloc(self.pool_size)
                self.free_blocks = [(0, self.pool_size)]
                self.initialized = True
                logger.info(f"Memory pool initialized: {self.pool_size / (1024**3):.2f}GB")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize memory pool: {e}")
            return False
    
    def allocate(self, size: int) -> Optional[int]:
        """Allocate memory from the pool"""
        if not self.initialized:
            return None
            
        # Align size to 256 bytes for better performance
        aligned_size = ((size + 255) // 256) * 256
        
        with self.lock:
            # Find a suitable free block
            for i, (offset, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # Allocate from this block
                    self.allocated_blocks[offset] = aligned_size
                    
                    # Update free blocks
                    if block_size == aligned_size:
                        # Exact fit, remove the block
                        del self.free_blocks[i]
                    else:
                        # Split the block
                        self.free_blocks[i] = (offset + aligned_size, block_size - aligned_size)
                    
                    return int(self.pool_ptr) + offset
            
            logger.warning(f"Memory pool exhausted, requested {aligned_size} bytes")
            return None
    
    def deallocate(self, ptr: int):
        """Deallocate memory back to the pool"""
        if not self.initialized:
            return
            
        offset = ptr - int(self.pool_ptr)
        
        with self.lock:
            if offset not in self.allocated_blocks:
                logger.warning(f"Attempt to deallocate unallocated memory at offset {offset}")
                return
                
            size = self.allocated_blocks.pop(offset)
            
            # Add back to free blocks and merge adjacent blocks
            self.free_blocks.append((offset, size))
            self.free_blocks.sort()
            
            # Merge adjacent free blocks
            merged_blocks = []
            for block_offset, block_size in self.free_blocks:
                if merged_blocks and merged_blocks[-1][0] + merged_blocks[-1][1] == block_offset:
                    # Merge with previous block
                    merged_blocks[-1] = (merged_blocks[-1][0], merged_blocks[-1][1] + block_size)
                else:
                    merged_blocks.append((block_offset, block_size))
            
            self.free_blocks = merged_blocks
    
    def cleanup(self):
        """Clean up the memory pool"""
        with self.lock:
            if self.pool_ptr:
                try:
                    self.pool_ptr.free()
                except:
                    pass
                self.pool_ptr = None
            self.allocated_blocks.clear()
            self.free_blocks.clear()
            self.initialized = False


class CudaContextManager:
    """Manages separate CUDA contexts for TensorRT and PyTorch operations"""
    
    def __init__(self):
        self.tensorrt_context = None
        self.pytorch_context = None
        self.device_id = 0
        self.memory_pools = {}
        self.health_status = {
            'tensorrt_healthy': True,
            'pytorch_healthy': True,
            'last_check': time.time(),
            'error_count': 0,
            'last_error': None
        }
        self.lock = threading.Lock()
        self._cleanup_callbacks = []
        
    def initialize(self, device_id: int = 0) -> bool:
        """Initialize separate CUDA contexts"""
        if not PYCUDA_AVAILABLE:
            logger.warning("PyCUDA not available, context isolation disabled")
            return False
            
        try:
            with self.lock:
                self.device_id = device_id
                
                # Initialize device
                cuda.init()
                device = cuda.Device(device_id)
                
                # Create separate contexts
                self.tensorrt_context = device.make_context(flags=cuda.ctx_flags.SCHED_AUTO)
                self.pytorch_context = device.make_context(flags=cuda.ctx_flags.SCHED_AUTO)
                
                # Initialize memory pools for each context
                self.memory_pools['tensorrt'] = MemoryPool(2 * 1024 * 1024 * 1024)  # 2GB
                self.memory_pools['pytorch'] = MemoryPool(1 * 1024 * 1024 * 1024)   # 1GB
                
                # Switch to TensorRT context and initialize its pool
                self.tensorrt_context.push()
                tensorrt_init = self.memory_pools['tensorrt'].initialize()
                self.tensorrt_context.pop()
                
                # Switch to PyTorch context and initialize its pool
                self.pytorch_context.push()
                pytorch_init = self.memory_pools['pytorch'].initialize()
                self.pytorch_context.pop()
                
                if tensorrt_init and pytorch_init:
                    logger.info("CUDA contexts and memory pools initialized successfully")
                    return True
                else:
                    logger.error("Failed to initialize memory pools")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to initialize CUDA contexts: {e}")
            self.health_status['tensorrt_healthy'] = False
            self.health_status['last_error'] = str(e)
            return False
    
    @contextmanager
    def tensorrt_context_scope(self):
        """Context manager for TensorRT operations with error handling"""
        if not self.tensorrt_context or not self.health_status['tensorrt_healthy']:
            raise RuntimeError("TensorRT context not available or unhealthy")
            
        try:
            with self.lock:
                self.tensorrt_context.push()
            yield self.memory_pools['tensorrt']
            
        except Exception as e:
            logger.error(f"Error in TensorRT context: {e}")
            self.health_status['tensorrt_healthy'] = False
            self.health_status['error_count'] += 1
            self.health_status['last_error'] = str(e)
            raise
            
        finally:
            try:
                with self.lock:
                    self.tensorrt_context.pop()
            except:
                pass
    
    @contextmanager
    def pytorch_context_scope(self):
        """Context manager for PyTorch operations with error handling"""
        if not self.pytorch_context or not self.health_status['pytorch_healthy']:
            raise RuntimeError("PyTorch context not available or unhealthy")
            
        try:
            with self.lock:
                self.pytorch_context.push()
            yield self.memory_pools['pytorch']
            
        except Exception as e:
            logger.error(f"Error in PyTorch context: {e}")
            self.health_status['pytorch_healthy'] = False
            self.health_status['error_count'] += 1
            self.health_status['last_error'] = str(e)
            raise
            
        finally:
            try:
                with self.lock:
                    self.pytorch_context.pop()
            except:
                pass
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health status of CUDA contexts"""
        try:
            current_time = time.time()
            
            # Test TensorRT context if it's supposed to be healthy
            if self.health_status['tensorrt_healthy'] and self.tensorrt_context:
                try:
                    with self.tensorrt_context_scope():
                        # Simple memory test
                        test_ptr = cuda.mem_alloc(1024)
                        test_ptr.free()
                except Exception as e:
                    logger.warning(f"TensorRT context health check failed: {e}")
                    self.health_status['tensorrt_healthy'] = False
                    self.health_status['last_error'] = str(e)
            
            # Test PyTorch context if it's supposed to be healthy
            if self.health_status['pytorch_healthy'] and self.pytorch_context:
                try:
                    with self.pytorch_context_scope():
                        # Simple memory test
                        test_ptr = cuda.mem_alloc(1024)
                        test_ptr.free()
                except Exception as e:
                    logger.warning(f"PyTorch context health check failed: {e}")
                    self.health_status['pytorch_healthy'] = False
                    self.health_status['last_error'] = str(e)
            
            self.health_status['last_check'] = current_time
            return self.health_status.copy()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.health_status['tensorrt_healthy'] = False
            self.health_status['pytorch_healthy'] = False
            self.health_status['last_error'] = str(e)
            return self.health_status.copy()
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover from CUDA context corruption"""
        logger.info("Attempting CUDA context recovery...")
        
        try:
            with self.lock:
                # Clean up existing contexts
                self.cleanup()
                
                # Wait a bit for cleanup
                time.sleep(0.5)
                
                # Reinitialize
                if self.initialize(self.device_id):
                    self.health_status['error_count'] = 0
                    self.health_status['last_error'] = None
                    logger.info("CUDA context recovery successful")
                    return True
                else:
                    logger.error("CUDA context recovery failed")
                    return False
                    
        except Exception as e:
            logger.error(f"CUDA context recovery failed: {e}")
            return False
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a callback to be called during cleanup"""
        self._cleanup_callbacks.append(weakref.ref(callback))
    
    def cleanup(self):
        """Clean up all CUDA contexts and memory pools"""
        logger.info("Cleaning up CUDA contexts...")
        
        # Call cleanup callbacks
        for callback_ref in self._cleanup_callbacks:
            callback = callback_ref()
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
        
        # Clean up memory pools
        for pool in self.memory_pools.values():
            try:
                pool.cleanup()
            except Exception as e:
                logger.warning(f"Memory pool cleanup failed: {e}")
        
        # Clean up contexts
        try:
            if self.tensorrt_context:
                self.tensorrt_context.detach()
                self.tensorrt_context = None
        except:
            pass
            
        try:
            if self.pytorch_context:
                self.pytorch_context.detach()
                self.pytorch_context = None
        except:
            pass
        
        self.memory_pools.clear()
        self.health_status = {
            'tensorrt_healthy': False,
            'pytorch_healthy': False,
            'last_check': time.time(),
            'error_count': 0,
            'last_error': None
        }


# Global context manager instance
_global_context_manager = None
_context_lock = threading.Lock()


def get_context_manager() -> CudaContextManager:
    """Get the global CUDA context manager instance"""
    global _global_context_manager
    
    with _context_lock:
        if _global_context_manager is None:
            _global_context_manager = CudaContextManager()
            if not _global_context_manager.initialize():
                logger.warning("Failed to initialize CUDA context manager")
        
        return _global_context_manager


def cleanup_global_context():
    """Clean up the global context manager"""
    global _global_context_manager
    
    with _context_lock:
        if _global_context_manager:
            _global_context_manager.cleanup()
            _global_context_manager = None


# Register cleanup on module exit
import atexit
atexit.register(cleanup_global_context)