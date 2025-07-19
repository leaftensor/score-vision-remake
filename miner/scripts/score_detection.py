import json
from pathlib import Path
import asyncio
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse, GSRChallenge, ChallengeType
from datetime import datetime
import cv2
import numpy as np
from validator.evaluation.evaluation import COLORS
import tempfile
import requests
import time
import math

# Set the test video URL and input file path
TEST_VIDEO_URL = "https://scoredata.me/2025_06_18/2025_06_18_d49f45ff/2025_06_18_d49f45ff_195945e20e9e4325b51ab84ff134c7_dcb7b85f.mp4"
INPUT_SCORE_FILE = r"C:\Users\longp\Documents\GitHub\score-vision\miner\test_outputs\pipeline_results_1752860308.json"

# Constants from validator config
MAX_PROCESSING_TIME = 15.0

def draw_annotations(frame, detections):
    out = frame.copy()
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
    for kp in detections.get("keypoints", []):
        if kp and len(kp) == 2 and kp[0] != 0 and kp[1] != 0:
            cv2.circle(out, (int(kp[0]), int(kp[1])), 5, COLORS["keypoint"], -1)
    return out

def download_video(url):
    start_time = time.time()
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        elapsed = time.time() - start_time
        print(f"Video downloaded to: {tmp.name} in {elapsed:.2f} seconds")
        return tmp.name
    else:
        raise Exception(f"Failed to download video: {resp.status_code}") 

def calculate_final_score(keypoint_score, bbox_score):
    """
    Calculate final score based on keypoint and bbox scores.
    Updated formula to match validator's new scoring system:
    - Quality Score = bbox_score * 0.75 + keypoint_score * 0.25
    - This reflects the validator's emphasis on bbox accuracy
    """
    quality_score = (bbox_score * 0.75) + (keypoint_score * 0.25)
    return quality_score

def calculate_total_final_score(quality_score, speed_score):
    """
    Calculate the total final score combining quality and speed.
    Formula from validator: final_score = quality_score * 0.65 + speed_score * 0.35
    """
    return (quality_score * 0.65) + (speed_score * 0.35)

def calculate_speed_score(processing_time: float, min_time: float, max_time: float) -> float:
    """
    Calculate speed score using validator's method.
    Based on validator/evaluation/calculate_score.py - calculate_speed_score function
    """
    if processing_time <= 0:
        return 0.0
    
    # Check if processing time exceeds maximum allowed
    if processing_time >= MAX_PROCESSING_TIME:
        return 0.0
    
    # If all processing times are the same, return 1.0
    if max_time == min_time:
        return 1.0
    
    # Normalize processing time (0 = fastest, 1 = slowest within valid range)
    time_range = max_time - min_time
    normalized_time = (processing_time - min_time) / time_range if time_range > 0 else 0
    
    # Speed score: faster is better (1 - normalized_time)
    speed_score = 1.0 - normalized_time
    
    return max(0.0, min(1.0, speed_score))

def calculate_speed_score_validator(processing_time: float, min_time: float = 0.0, max_time: float = 0.0) -> float:
    """
    Calculate speed score using validator's exponential scaling method.
    Based on validator/evaluation/calculate_score.py
    DEPRECATED: Use calculate_speed_score instead for validator compatibility
    """
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
    # Using exponential decay with base e
    exp_score = math.exp(-5 * normalized_time)  # -5 controls steepness of decay
    
    return max(0.0, min(1.0, exp_score))  # Ensure score stays in 0-1 range

def calculate_speed_score_fps(processing_time, total_frames):
    """
    Calculate speed score based on FPS (alternative method).
    Higher score for faster processing.
    """
    if processing_time <= 0 or total_frames <= 0:
        return 0.0
    
    # Calculate frames per second
    fps = total_frames / processing_time
    
    # Score based on FPS (higher FPS = higher score)
    # Target: 30 FPS = 100 score, 15 FPS = 50 score, 5 FPS = 0 score
    if fps >= 30:
        return 100.0
    elif fps >= 15:
        return 50.0 + (fps - 15) * 3.33  # Linear interpolation
    elif fps >= 5:
        return (fps - 5) * 5.0  # Linear interpolation
    else:
        return 0.0

async def main():
    start_time = time.time()
    
    # Load detection output from the specified file
    with open(INPUT_SCORE_FILE, "r") as f:
        detection = json.load(f)

    # Download video if it's a URL
    video_path = TEST_VIDEO_URL
    if video_path.startswith("http"):  # Download to temp file
        video_path = download_video(video_path)
    
    # Visualize and save video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out_path = 'output_annotated.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_idx = 0
    frames = detection["frames"] if "frames" in detection else detection
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
    print(f"Annotated video saved to {out_path}")

    # Prepare GSRResponse
    response = GSRResponse(
        challenge_id="test-challenge",
        frames=detection["frames"] if "frames" in detection else detection,
        processing_time=detection.get("processing_time", None)
    )

    print(f"Video path: {video_path}")
    # Prepare dummy GSRChallenge (video path must match the one used for detection)
    challenge = GSRChallenge(
        challenge_id="test-challenge",
        type=ChallengeType.GSR,
        created_at=datetime.now(),
        video_url=video_path
    )

    # Instantiate validator (API key can be dummy if not using OpenAI features)
    validator = GSRValidator(openai_api_key="", validator_hotkey="dummy_hotkey")

    # Select frames to validate (e.g., all frames)
    frames_to_validate = [int(k) for k in response.frames.keys()]
    selected_frames_id_bbox = frames_to_validate

    # Run scoring
    result = await validator.evaluate_response(
        response=response,
        challenge=challenge,
        video_path=Path(challenge.video_url),
        video_width=1920,
        video_height=1080,
        frames_to_validate=frames_to_validate,
        selected_frames_id_bbox=selected_frames_id_bbox
    )

    # Calculate additional scores
    total_time = time.time() - start_time
    total_frames = len(frames_to_validate)
    processing_time = 5
    # detection.get("processing_time", total_time)
    
    # Extract keypoint score from feedback
    keypoints_final_score = result.feedback.get("keypoints_final_score", 0.0) if isinstance(result.feedback, dict) else 0.0
    bbox_score = result.score  # This is the bbox score from evaluate_bboxes
    
    # Calculate quality score using new validator formula
    quality_score = calculate_final_score(keypoints_final_score, bbox_score)
    
    # Calculate speed score using validator method
    # For single test, we simulate min/max times for demonstration
    min_time = max(processing_time * 0.5, 1.0)  # Simulated best case
    max_time = min(processing_time * 1.5, MAX_PROCESSING_TIME)  # Simulated worst case
    speed_score_validator = calculate_speed_score(processing_time, min_time, max_time)
    
    # Calculate total final score (quality + speed)
    total_final_score = calculate_total_final_score(quality_score, speed_score_validator)
    
    # Calculate FPS-based score for comparison
    speed_score_fps = calculate_speed_score_fps(processing_time, total_frames)
    
    # Extract keypoint component details from feedback if available
    feedback_dict = result.feedback if isinstance(result.feedback, dict) else {}
    keypoint_components = feedback_dict.get("components", {})
    
    # Print in validator's exact format
    print(f"bbox_score: {bbox_score}, keypoints_final_score: {keypoints_final_score}")
    
    # Print keypoint component scores in validator format
    print(f"\nComponent Scores:")
    print(f"Keypoint Score: {keypoint_components.get('avg_keypoint_score', 0.0):.2f} (weight: 0.25)")
    print(f"Point on line Score: {keypoint_components.get('mean_on_line', 0.0)*100:.2f} (weight: 0.4)")
    print(f"Mean inside Score: {keypoint_components.get('mean_inside', 0.0)*100:.2f} (weight: 0.2)")
    print(f"Keypoint Stability: {0.0:.2f} (weight: 0.05)")
    print(f"Player final score: {keypoint_components.get('player_score', 0.0):.2f} (weight: 0.1)")
    
    # Print validator-style detailed logs
    print(f"\nProcessing times: Min time: {min_time}, Max time: {max_time}")
    print(f"Final score for response test-challenge: {total_final_score}")
    
    # Print response scoring details in validator format
    print(f"Response test-challenge scoring details:")
    print(f"  - Quality score: {quality_score:.3f}")
    print(f"  - Speed score: {speed_score_validator:.3f}")
    print(f"  - Final score: {total_final_score:.3f}")
    
    # Print additional validator-style output
    print(f"\nTotal frames processed: {total_frames}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"FPS: {total_frames/processing_time:.2f}")
    print(f"MAX_PROCESSING_TIME: {MAX_PROCESSING_TIME} seconds")
    
    # Print detailed breakdown for comparison
    print("\n" + "="*70)
    print("DETAILED SCORING BREAKDOWN (For Analysis)")
    print("="*70)
    print("COMPONENT SCORES:")
    print(f"  BBox Score: {bbox_score:.4f}")
    print(f"  Keypoint Score: {keypoints_final_score:.4f}")
    print(f"  Quality Score (75% bbox + 25% keypoint): {quality_score:.4f}")
    print(f"  Speed Score: {speed_score_validator:.4f}")
    print("-"*70)
    print("FINAL SCORES:")
    print(f"  Total Score (65% quality + 35% speed): {total_final_score:.4f}")
    print(f"  Speed Score (FPS method, for comparison): {speed_score_fps:.2f}/100")
    print("-"*70)
    print("VALIDATOR WEIGHT BREAKDOWN:")
    print("  Quality Score = BBox(75%) + Keypoints(25%)")
    print("    ├─ BBox: Object detection accuracy")
    print("    └─ Keypoints: Field alignment + spatial accuracy")
    print("      ├─ keypoint_score: 25% - Basic keypoint quality")
    print("      ├─ mean_on_line: 40% - Distance to pitch lines")
    print("      ├─ mean_inside: 20% - Field projection accuracy")
    print("      ├─ keypoint_stability: 5% - Temporal consistency")
    print("      └─ player_final_score: 10% - Player movement validation")
    print("  Total Score = Quality(65%) + Speed(35%)")
    print("-"*70)
    print(f"Frame scores: {result.frame_scores}")
    print(f"Feedback: {result.feedback}")
    print("="*70)
    
    # Save detailed results to file with updated structure
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "video_path": video_path,
        "processing_metrics": {
            "processing_time": processing_time,
            "total_frames": total_frames,
            "fps": total_frames/processing_time,
            "max_processing_time": MAX_PROCESSING_TIME
        },
        "component_scores": {
            "bbox_score": bbox_score,
            "keypoint_score": keypoints_final_score,
            "quality_score": quality_score,
            "speed_score": speed_score_validator,
            "speed_score_fps": speed_score_fps
        },
        "final_scores": {
            "total_final_score": total_final_score,
            "quality_weight": 0.65,
            "speed_weight": 0.35
        },
        "validator_weights": {
            "quality_composition": {
                "bbox_weight": 0.75,
                "keypoint_weight": 0.25
            },
            "keypoint_components": {
                "keypoint_score": 0.25,
                "mean_on_line": 0.40,
                "mean_inside": 0.20,
                "keypoint_stability": 0.05,
                "player_final_score": 0.10
            }
        },
        "frame_scores": result.frame_scores,
        "feedback": result.feedback
    }
    
    with open("scoring_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: scoring_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 
