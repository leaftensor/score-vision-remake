#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from pathlib import Path
import time
from loguru import logger
from typing import List, Dict, Union
import cv2
import numpy as np
import tempfile
import requests
import math
from datetime import datetime

# Add miner directory to path
miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

# Import miner modules
from utils.model_manager import ModelManager
from utils.video_downloader import download_video
from endpoints.soccer import process_soccer_video
from utils.device import get_optimal_device
from scripts.download_models import download_models

# Import validator modules
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse, GSRChallenge, ChallengeType
from validator.evaluation.evaluation import COLORS

# Configuration
TEST_VIDEO_URL = "https://scoredata.me/2025_06_18/2025_06_18_d49f45ff/2025_06_18_d49f45ff_195945e20e9e4325b51ab84ff134c7_dcb7b85f.mp4"
MAX_PROCESSING_TIME = 15.0

def optimize_coordinates(coords: List[float]) -> List[float]:
    """Optimize coordinate precision to 2 decimal places."""
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Filter out invalid keypoints (0,0 coordinates)."""
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

def optimize_frame_data(frame_data: Dict) -> Dict:
    """Optimize frame data for storage and transmission."""
    optimized_data = {}
    
    if "objects" in frame_data:
        optimized_data["objects"] = []
        for obj in frame_data["objects"]:
            optimized_obj = obj.copy()
            if "bbox" in obj:
                optimized_obj["bbox"] = optimize_coordinates(obj["bbox"])
            optimized_data["objects"].append(optimized_obj)
    
    if "keypoints" in frame_data:
        optimized_data["keypoints"] = filter_keypoints(frame_data["keypoints"])
    
    return optimized_data

def optimize_result_data(result: Dict[str, Union[Dict, List, float, str]]) -> Dict[str, Union[Dict, List, float, str]]:
    """Optimize entire result data structure."""
    optimized_result = result.copy()
    
    if "frames" in result:
        frames = result["frames"]
        
        if isinstance(frames, list):
            optimized_frames = {}
            for i, frame_data in enumerate(frames):
                if frame_data:
                    optimized_frames[str(i)] = optimize_frame_data(frame_data)
        elif isinstance(frames, dict):
            optimized_frames = {}
            for frame_num, frame_data in frames.items():
                if frame_data:
                    optimized_frames[str(frame_num)] = optimize_frame_data(frame_data)
        else:
            logger.warning(f"Unexpected frames data type: {type(frames)}")
            optimized_frames = frames
            
        optimized_result["frames"] = optimized_frames
    
    if "processing_time" in result:
        optimized_result["processing_time"] = round(float(result["processing_time"]), 2)
    
    return optimized_result

def draw_annotations(frame, detections):
    """Draw bounding boxes and keypoints on frame for visualization."""
    out = frame.copy()
    
    # Draw bounding boxes
    for obj in detections.get("objects", []):
        (x1, y1, x2, y2) = map(int, obj["bbox"])
        cid = obj["class_id"]
        if cid == 0:
            color = COLORS["ball"]
        elif cid == 1:
            color = COLORS["goalkeeper"]
        elif cid == 3:
            color = COLORS["referee"]
        else:
            color = COLORS["player"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, str(cid), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw keypoints
    for kp in detections.get("keypoints", []):
        if kp and len(kp) == 2 and kp[0] != 0 and kp[1] != 0:
            cv2.circle(out, (int(kp[0]), int(kp[1])), 5, COLORS["keypoint"], -1)
    
    return out

def calculate_final_score(keypoint_score, bbox_score):
    """Calculate final score based on keypoint and bbox scores."""
    return 0.1 * keypoint_score + 0.9 * bbox_score

def calculate_speed_score_validator(processing_time: float, min_time: float = 0.0, max_time: float = 0.0) -> float:
    """Calculate speed score using validator's exponential scaling method."""
    if processing_time <= 0:
        return 0.0
    
    # If min_time and max_time are not provided, use current time as reference
    if min_time == 0.0:
        min_time = processing_time
    if max_time == 0.0:
        max_time = processing_time
    
    # Check if processing time exceeds maximum allowed
    if processing_time >= MAX_PROCESSING_TIME:
        return 0.0
    
    # If all times are the same, give full score
    if max_time == min_time:
        return 1.0
        
    # Normalize time to 0-1 range
    normalized_time = (processing_time - min_time) / (min(max_time, MAX_PROCESSING_TIME) - min_time)
    
    # Apply exponential scaling to more aggressively reward faster times
    exp_score = math.exp(-5 * normalized_time)
    
    return max(0.0, min(1.0, exp_score))

def calculate_speed_score_fps(processing_time, total_frames):
    """Calculate speed score based on FPS."""
    if processing_time <= 0 or total_frames <= 0:
        return 0.0
    
    fps = total_frames / processing_time
    
    if fps >= 30:
        return 100.0
    elif fps >= 15:
        return 50.0 + (fps - 15) * 3.33
    elif fps >= 5:
        return (fps - 5) * 5.0
    else:
        return 0.0

def create_annotated_video(video_path: str, detection_result: Dict, output_path: str = "output_annotated.mp4"):
    """Create annotated video with bounding boxes and keypoints."""
    logger.info("Creating annotated video...")
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    frames = detection_result["frames"] if "frames" in detection_result else detection_result
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det = frames.get(str(frame_idx), {"objects": [], "keypoints": []})
        vis = draw_annotations(frame, det)
        writer.write(vis)
        frame_idx += 1
    
    cap.release()
    writer.release()
    logger.info(f"Annotated video saved to {output_path}")

async def evaluate_detection(detection_result: Dict, video_path: str):
    """Evaluate detection results using validator."""
    logger.info("Starting detection evaluation...")
    
    # Prepare GSRResponse
    response = GSRResponse(
        challenge_id="test-challenge",
        frames=detection_result["frames"] if "frames" in detection_result else detection_result,
        processing_time=detection_result.get("processing_time", None)
    )

    # Prepare dummy GSRChallenge
    challenge = GSRChallenge(
        challenge_id="test-challenge",
        type=ChallengeType.GSR,
        created_at=datetime.now(),
        video_url=video_path
    )

    # Instantiate validator
    validator = GSRValidator(openai_api_key="", validator_hotkey="dummy_hotkey")

    # Select frames to validate
    frames_to_validate = [int(k) for k in response.frames.keys()]
    selected_frames_id_bbox = frames_to_validate

    # Run scoring
    result = await validator.evaluate_response(
        response=response,
        challenge=challenge,
        video_path=Path(challenge.video_url),
        frames_to_validate=frames_to_validate,
        selected_frames_id_bbox=selected_frames_id_bbox
    )
    
    return result

def print_detailed_results(result, processing_time, total_frames, detection_result, video_path):
    """Print comprehensive scoring results."""
    # Extract scores
    keypoint_final_score = result.feedback.get("keypoints_final_score", 0.0) if isinstance(result.feedback, dict) else 0.0
    bbox_score = result.score
    final_score = calculate_final_score(keypoint_final_score, bbox_score)
    
    # Calculate speed scores
    speed_score_validator = calculate_speed_score_validator(processing_time)
    speed_score_fps = calculate_speed_score_fps(processing_time, total_frames)
    
    # Print results
    print("\n" + "="*80)
    print("COMPLETE PIPELINE RESULTS")
    print("="*80)
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Total Frames: {total_frames}")
    print(f"FPS: {total_frames/processing_time:.2f}")
    print(f"MAX_PROCESSING_TIME: {MAX_PROCESSING_TIME} seconds")
    print("-"*80)
    print(f"Speed Score (Validator method): {speed_score_validator:.4f}")
    print(f"Speed Score (FPS method): {speed_score_fps:.2f}/100")
    print("-"*80)
    print(f"BBox Score: {bbox_score:.4f}")
    print(f"Keypoint Score: {keypoint_final_score:.4f}")
    print(f"Final Score: {final_score:.4f}")
    print("-"*80)
    print(f"Frame scores: {result.frame_scores}")
    print(f"Feedback: {result.feedback}")
    print("="*80)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "video_path": video_path,
        "processing_time": processing_time,
        "total_frames": total_frames,
        "fps": total_frames/processing_time,
        "max_processing_time": MAX_PROCESSING_TIME,
        "speed_score_validator": speed_score_validator,
        "speed_score_fps": speed_score_fps,
        "bbox_score": bbox_score,
        "keypoint_score": keypoint_final_score,
        "final_score": final_score,
        "frame_scores": result.frame_scores,
        "feedback": result.feedback
    }

async def main():
    """Main pipeline function that combines detection and evaluation."""
    try:
        logger.info("Starting complete video processing pipeline")
        pipeline_start_time = time.time()
        
        # Step 1: Download and prepare models
        logger.info("Step 1: Checking for required models...")
        download_models()
        
        # Step 2: Download video
        logger.info(f"Step 2: Downloading test video from {TEST_VIDEO_URL}")
        video_path = await download_video(TEST_VIDEO_URL)
        logger.info(f"Video downloaded to {video_path}")
        
        try:
            # Step 3: Initialize models and device
            device = get_optimal_device()
            logger.info(f"Using device: {device}")
            
            model_manager = ModelManager(device=device)
            logger.info("Loading models...")
            model_manager.load_all_models()
            logger.info("Models loaded successfully")
            
            # Step 4: Process video (detection)
            logger.info("Step 3: Starting video processing and detection...")
            detection_start_time = time.time()
            
            result = await process_soccer_video(
                video_path=str(video_path),
                model_manager=model_manager
            )
            
            detection_time = time.time() - detection_start_time
            logger.info(f"Detection completed in {detection_time:.2f} seconds")
            
            # Step 5: Optimize results
            logger.info("Step 4: Optimizing detection results...")
            optimized_result = optimize_result_data(result)
            
            # Step 6: Save detection results
            output_dir = Path(__file__).parent.parent / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            detection_output_file = output_dir / f"pipeline_results_{int(time.time())}.json"
            
            result_json = json.dumps(optimized_result)
            data_size = len(result_json) / 1024
            logger.info(f"Detection result data size: {data_size:.2f} KB")
            
            with open(detection_output_file, "w") as f:
                f.write(result_json)
            
            logger.info(f"Detection results saved to: {detection_output_file}")
            
            # Step 7: Create annotated video
            logger.info("Step 5: Creating annotated video...")
            annotated_video_path = output_dir / f"annotated_video_{int(time.time())}.mp4"
            create_annotated_video(str(video_path), optimized_result, str(annotated_video_path))
            
            # Step 8: Evaluate detection results
            logger.info("Step 6: Evaluating detection results...")
            evaluation_result = await evaluate_detection(optimized_result, str(video_path))
            
            # Step 9: Calculate and display final results
            logger.info("Step 7: Calculating final scores...")
            total_frames = len(optimized_result["frames"])
            processing_time = optimized_result.get("processing_time", detection_time)
            
            results_summary = print_detailed_results(
                evaluation_result, 
                processing_time, 
                total_frames, 
                optimized_result,
                str(video_path)
            )
            
            # Step 10: Save comprehensive results
            comprehensive_results_file = output_dir / f"comprehensive_results_{int(time.time())}.json"
            with open(comprehensive_results_file, "w") as f:
                json.dump(results_summary, f, indent=2)
            
            # Final summary
            total_pipeline_time = time.time() - pipeline_start_time
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Total pipeline time: {total_pipeline_time:.2f} seconds")
            logger.info(f"Detection time: {detection_time:.2f} seconds")
            logger.info(f"Evaluation time: {total_pipeline_time - detection_time:.2f} seconds")
            logger.info(f"Total frames processed: {total_frames}")
            logger.info(f"Detection FPS: {total_frames/detection_time:.2f}")
            logger.info(f"Final Score: {results_summary['final_score']:.4f}")
            logger.info(f"Detection results: {detection_output_file}")
            logger.info(f"Annotated video: {annotated_video_path}")
            logger.info(f"Comprehensive results: {comprehensive_results_file}")
            logger.info("="*80)
            
        finally:
            model_manager.clear_cache()
            
    finally:
        try:
            video_path.unlink()
            logger.info("Cleaned up temporary video file")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 