import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger
import cv2
import random

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video

from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
from supervision.keypoint.core import KeyPoints
from validator.evaluation.bbox_clip import batch_classify_rois as orig_batch_classify_rois, BoundingBoxObject, OBJECT_ID_TO_ENUM, BBoxScore, extract_regions_of_interest_from_image, evaluate_frame as evaluate_frame_filter
import torch
from validator.evaluation.bbox_clip import batch_evaluate_frame_filter

# Override batch_classify_rois để dùng GPU nếu có

def batch_classify_rois_gpu(regions_of_interest):
    from validator.evaluation.bbox_clip import data_processor, clip_model, BoundingBoxObject, OBJECT_ID_TO_ENUM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_inputs = data_processor(
        text=[key.value for key in BoundingBoxObject],
        images=regions_of_interest,
        return_tensors="pt",
        padding=True
    ).to(device)
    clip_model.to(device)
    with torch.no_grad():
        model_outputs = clip_model(**model_inputs)
        probabilities = model_outputs.logits_per_image.softmax(dim=1)
        object_ids = probabilities.argmax(dim=1)
    return [OBJECT_ID_TO_ENUM.get(object_id.item(), BoundingBoxObject.OTHER) for object_id in object_ids]

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Default configuration
DEFAULT_FILTER_TYPE = 'batch'
DEFAULT_FILTER_ENABLED = True

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

# --- Thay thế 2 hàm scale cứng bằng hàm scale động ---
def scale_bbox_between_sizes(bbox, from_size, to_size):
    """
    Scale bbox từ from_size (w, h) sang to_size (w, h).
    bbox: [x1, y1, x2, y2]
    from_size, to_size: (width, height)
    """
    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]
    x1, y1, x2, y2 = bbox
    return [
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y
    ]

def scale_keypoints_between_sizes(keypoints, from_size, to_size):
    """
    Scale keypoints từ from_size (w, h) sang to_size (w, h).
    keypoints: list of [x, y]
    """
    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]
    return [[x * scale_x, y * scale_y] for x, y in keypoints]

async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
    filter_type: str = 'batch',  # Thêm tham số để chọn logic filter
    filter_enabled: bool = True
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data (batch optimized)."""
    import math
    timings = {}
    start_time = time.time()
    try:
        
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )

        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )

        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        tracker = ByteTrack()
        tracking_data = {"frames": []}

        # 1. Load toàn bộ frame vào RAM
        t_load_start = time.time()
        frames = []
        empty_object_frame_ids = []  # Lưu id các frame không có object
        async for frame_number, frame in video_processor.stream_frames(video_path):
            frames.append((frame_number, frame))
        t_load_end = time.time()
        timings['load_frames'] = t_load_end - t_load_start

        batch_size = 10  # Có thể điều chỉnh tùy GPU
        total_frames = len(frames)
        frames_with_objects = set()
        # 2. Detect object trên frame đã resize
        t_infer_start = time.time()
        for i in range(0, total_frames, batch_size):
            batch_tuples = frames[i:i+batch_size]
            batch_indices = [t[0] for t in batch_tuples]
            batch_frames = [t[1] for t in batch_tuples]
            orig_height, orig_width = batch_frames[0].shape[:2]
            resize_width, resize_height = 640, 360
            batch_frames_resized = [cv2.resize(frame, (resize_width, resize_height)) for frame in batch_frames]
            player_results = player_model(batch_frames_resized, verbose=False)
            for j, (frame_number, frame) in enumerate(batch_tuples):
                player_result = player_results[j]
                detections = Detections.from_ultralytics(player_result)
                detections = tracker.update_with_detections(detections)
                tracker_ids = getattr(detections, 'tracker_id', None)
                xyxys = getattr(detections, 'xyxy', None)
                class_ids = getattr(detections, 'class_id', None)
                objects = []
                if (
                    tracker_ids is not None and xyxys is not None and class_ids is not None
                    and hasattr(tracker_ids, '__len__') and hasattr(xyxys, '__len__') and hasattr(class_ids, '__len__')
                    and len(tracker_ids) == len(xyxys) == len(class_ids)
                ):
                    for tracker_id, bbox, class_id in zip(tracker_ids.tolist(), xyxys.tolist(), class_ids.tolist()):
                        # Không scale bbox ở đây, giữ bbox theo frame đã resize
                        objects.append({
                            "id": int(tracker_id),
                            "bbox": [float(x) for x in bbox],
                            "class_id": int(class_id)
                        })
                
                if objects:
                    frames_with_objects.add(frame_number)
                else:
                    empty_object_frame_ids.append(frame_number)
                frame_data = {
                    "frame_number": int(frame_number),
                    "objects": objects,
                    "frame": frame,  # frame đã resize
                    "orig_shape": (orig_width, orig_height),
                    "resize_shape": (resize_width, resize_height)
                }
                tracking_data["frames"].append(frame_data)
        t_infer_end = time.time()
        timings['batch_inference'] = t_infer_end - t_infer_start

        # 4. Scale bbox về kích thước gốc trước khi filter CLIP
        for frame_data in tracking_data["frames"]:
            if not isinstance(frame_data["objects"], list):
                frame_data["objects"] = []
            orig_width, orig_height = frame_data["orig_shape"]
            resize_width, resize_height = frame_data["resize_shape"]
            for obj in frame_data["objects"]:
                obj["bbox"] = scale_bbox_between_sizes(obj["bbox"], (resize_width, resize_height), (orig_width, orig_height))
        # 5. CLIP-based filtering trên frame gốc và bbox gốc
        t_clip_start = time.time()
        def filter_objects_clip_frame(args):
            frame_data, frame_orig = args
            bboxes = frame_data.get("objects", [])
            if not bboxes:
                return frame_data
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    keep = evaluate_frame_filter(
                        frame_id=frame_data["frame_number"],
                        image_array=frame_orig,  # frame gốc
                        bboxes=bboxes       # bbox đã scale về gốc
                    )
                    if not isinstance(keep, list):
                        keep = []
                    frame_data["objects"] = keep
                    return frame_data
                except RuntimeError as e:
                    if "Already borrowed" in str(e):
                        if attempt < max_retries - 1:
                            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        else:
                            print("Skip frame {} due to tokenizer lock after {} retries".format(frame_data['frame_number'], max_retries))
                            frame_data["objects"] = []
                            return frame_data
                    else:
                        raise
            return frame_data
        import concurrent.futures
        if filter_type == 'batch':
            frames = tracking_data["frames"]
            # Lấy ảnh gốc đúng thứ tự frame_number
            images = []
            for fd in frames:
                if "frame" in fd:
                    images.append(fd["frame"])
                else:
                    raise ValueError(f"Frame {fd['frame_number']} missing 'frame' key for batch CLIP filtering.")
            filtered_frames = batch_evaluate_frame_filter(frames, images, enable_class_limits=False)
            for i, frame_data in enumerate(filtered_frames):
                tracking_data["frames"][i]["objects"] = frame_data["objects"]
        else:
            args_list = [(fd, frames[fd["frame_number"]][1]) for fd in tracking_data["frames"]]  # frame gốc
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                tracking_data["frames"] = list(executor.map(filter_objects_clip_frame, args_list))
        t_clip_end = time.time()
        timings['clip_filtering'] = t_clip_end - t_clip_start
        
        # 7. Enhanced pitch detection and keypoint generation
        t_pitch_start = time.time()
        
        # Process pitch detection in batches to avoid model size limits
        max_batch_size = 10  # Model max batch size
        total_frames = len(tracking_data["frames"])
        
        for batch_start in range(0, total_frames, max_batch_size):
            batch_end = min(batch_start + max_batch_size, total_frames)
            batch_frames = tracking_data["frames"][batch_start:batch_end]
            
            # Collect ORIGINAL frames for this batch (better quality for pitch detection)
            batch_images = []
            for i, frame_data in enumerate(batch_frames):
                # Get original frame from the frames list
                frame_idx = batch_start + i
                if frame_idx < len(frames):
                    try:
                        frame_tuple = frames[frame_idx]
                        if isinstance(frame_tuple, (tuple, list)) and len(frame_tuple) >= 2:
                            # Extract frame from tuple/list structure
                            frame_number, original_frame = frame_tuple[0], frame_tuple[1]
                            batch_images.append(original_frame)  # Use original high-res frame
                        elif hasattr(frame_tuple, 'shape'):
                            # Direct frame array
                            batch_images.append(frame_tuple)
                        else:
                            # Fallback to resized frame if tuple structure is unexpected
                            batch_images.append(frame_data["frame"])
                    except Exception as e:
                        print(f"Error accessing original frame at index {frame_idx}: {e}")
                        # Fallback to resized frame
                        batch_images.append(frame_data["frame"])
                else:
                    # Fallback to resized frame if original not available
                    batch_images.append(frame_data["frame"])
            
            try:
                # Run pitch model on this batch
                print(f"Running pitch model on batch {batch_start}-{batch_end} with {len(batch_images)} images")
                
                pitch_results = pitch_model(batch_images, verbose=False)
                
                # Process keypoints for each frame in this batch (using simple working approach)
                for i, frame_data in enumerate(batch_frames):
                    # Extract pitch keypoints using the same working approach as soccer.py
                    pitch_result = pitch_results[i]
                    # print(f"Pitch result for frame {frame_data['frame_number']}: {pitch_result}")
                    # print(pitch_result.boxes)
                    try:
                        # Check if pitch model actually has keypoints (not just detection boxes)
                        if hasattr(pitch_result, 'keypoints') and pitch_result.keypoints is not None:
                            # Pitch model has real keypoints - use supervision to extract them
                            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
                            keypoints_list = keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else []
                            if keypoints_list:
                                print(f"  Successfully extracted {len(keypoints_list)} keypoints from pitch model")
                        else:
                            # Pitch model only has detection boxes, no keypoints
                            keypoints_list = []
                            print(f"  Pitch model has {pitch_result.boxes.shape[0] if hasattr(pitch_result, 'boxes') and pitch_result.boxes is not None else 0} detections but no keypoints")
                        
                    except Exception as e:
                        keypoints_list = []
                        print(f"Error extracting keypoints for frame {frame_data['frame_number']}: {e}")
                    
                    # Store keypoints
                    frame_data["keypoints"] = keypoints_list
                        
            except Exception as e:
                print(f"Batch pitch detection failed for batch {batch_start}-{batch_end}: {e}")
                # No fake keypoints - just empty lists for failed batch
                for frame_data in batch_frames:
                    frame_data["keypoints"] = []
                
        t_pitch_end = time.time()
        timings['pitch_detection'] = t_pitch_end - t_pitch_start
        # 8. Result aggregation/cleanup
        t_agg_start = time.time()
        for frame_data in tracking_data["frames"]:
            if "frame" in frame_data:
                del frame_data["frame"]
            if "orig_shape" in frame_data:
                del frame_data["orig_shape"]
            if "resize_shape" in frame_data:
                del frame_data["resize_shape"]
        t_agg_end = time.time()
        timings['result_aggregation'] = t_agg_end - t_agg_start

        total_frames = len(tracking_data["frames"])
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = float(processing_time)
        timings['total'] = processing_time

        # Print/log timings for each step
        print("--- Pipeline Step Timings (seconds) ---")
        for step, duration in timings.items():
            print("{}: {:.2f}s".format(step, duration))
        print("--------------------------------------")

        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            "Completed processing {} frames in {:.1f}s "
            "({:.2f} fps) on {} device".format(total_frames, processing_time, fps, model_manager.device)
        )
        return tracking_data
    except Exception as e:
        logger.error(e)
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            video_path = await download_video(video_url)
            try:
                # Ensure video_path is a string
                if isinstance(video_path, Path):
                    video_path = str(video_path)
                
                # Get filter configuration from request or use defaults
                filter_type = challenge_data.get("filter_type", DEFAULT_FILTER_TYPE)
                filter_enabled = challenge_data.get("filter_enabled", DEFAULT_FILTER_ENABLED)
                
                logger.info(f"Processing with filter: {filter_type} (enabled: {filter_enabled})")
                
                tracking_data = await process_soccer_video(
                    video_path,
                    model_manager,
                    filter_type=filter_type,
                    filter_enabled=filter_enabled
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")

# Create router with dependencies
router = APIRouter()

router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)

# Function removed - replaced with real pitch detection