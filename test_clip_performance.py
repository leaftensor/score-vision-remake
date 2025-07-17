#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple test without complex dependencies
def test_clip_performance():
    try:
        from validator.evaluation.bbox_clip import batch_evaluate_frame_filter
        
        # Create mock frames with realistic object counts
        frames = []
        images = []
        
        # Create 50 test frames with varying object counts
        np.random.seed(42)  # For reproducible results
        for i in range(50):
            num_objects = np.random.randint(5, 25)  # 5-25 objects per frame
            objects = []
            
            for j in range(num_objects):
                # Random bbox coordinates
                x1 = np.random.randint(0, 1600)
                y1 = np.random.randint(200, 800)  # Above y=150 threshold
                w = np.random.randint(20, 120)
                h = np.random.randint(50, 150)
                x2 = x1 + w
                y2 = y1 + h
                
                # Mix of class IDs
                class_id = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
                
                objects.append({
                    "id": j,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id
                })
            
            frames.append({
                "frame_number": i,
                "objects": objects
            })
            
            # Create a random RGB image (1920x1080)
            image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            images.append(image)
        
        print(f"Created {len(frames)} test frames with total {sum(len(f['objects']) for f in frames)} objects")
        
        # Test the performance with class limits enabled
        print("\n=== Testing with Class Limits Enabled ===")
        start_time = time.time()
        result_frames = batch_evaluate_frame_filter(frames, images, enable_class_limits=True)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Calculate bbox score (simple version)
        total_input_objects = sum(len(f['objects']) for f in frames)
        total_output_objects = sum(len(f['objects']) for f in result_frames)
        bbox_score = total_output_objects / total_input_objects if total_input_objects > 0 else 0
        
        print(f"\n=== CLIP Performance Test Results ===")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Target time: 7.0s")
        print(f"Status: {'✅ PASS' if processing_time <= 7.0 else '❌ FAIL'}")
        print(f"Input objects: {total_input_objects}")
        print(f"Output objects: {total_output_objects}")
        print(f"Bbox score: {bbox_score:.3f}")
        print(f"Bbox score target: 1.0")
        print(f"Bbox status: {'✅ PERFECT' if bbox_score >= 0.95 else '⚠️  NEEDS IMPROVEMENT'}")
        
        if processing_time <= 7.0 and bbox_score >= 0.95:
            print("\n🎉 SUCCESS: Both speed and accuracy targets achieved!")
        elif processing_time <= 7.0:
            print("\n⚡ Speed target achieved but bbox score needs improvement")
        elif bbox_score >= 0.95:
            print("\n🎯 Accuracy target achieved but speed needs improvement")
        else:
            print("\n❌ Both speed and accuracy need improvement")
            
        return processing_time, bbox_score
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed")
        return None, None
    except Exception as e:
        print(f"Error during test: {e}")
        return None, None

if __name__ == "__main__":
    test_clip_performance()