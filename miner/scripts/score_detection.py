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
INPUT_SCORE_FILE = r"C:\Users\longp\Documents\GitHub\score-vision\miner\test_outputs\pipeline_test_results_1752204348.json"

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
    Formula: final_score = 0.5 * keypoint_score + 0.5 * bbox_score
    """
    return 0.5 * keypoint_score + 0.5 * bbox_score

def calculate_speed_score_validator(processing_time: float, min_time: float = 0.0, max_time: float = 0.0) -> float:
    """
    Calculate speed score using validator's exponential scaling method.
    Based on validator/evaluation/calculate_score.py
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
        frames_to_validate=frames_to_validate,
        selected_frames_id_bbox=selected_frames_id_bbox
    )

    # Calculate additional scores
    total_time = time.time() - start_time
    total_frames = len(frames_to_validate)
    processing_time = 5
    # detection.get("processing_time", total_time)
    
    # Extract keypoint score from feedback
    keypoint_final_score = result.feedback.get("keypoints_final_score", 0.0) if isinstance(result.feedback, dict) else 0.0
    bbox_score = result.score  # This is the bbox score from evaluate_bboxes
    
    # Calculate final score
    final_score = calculate_final_score(keypoint_final_score, bbox_score)
    
    # Calculate speed scores using both methods
    speed_score_validator = calculate_speed_score_validator(processing_time)
    speed_score_fps = calculate_speed_score_fps(processing_time, total_frames)
    
    # Print detailed results
    print("\n" + "="*60)
    print("SCORING RESULTS")
    print("="*60)
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Total Frames: {total_frames}")
    print(f"FPS: {total_frames/processing_time:.2f}")
    print(f"MAX_PROCESSING_TIME: {MAX_PROCESSING_TIME} seconds")
    print("-"*60)
    print(f"Speed Score (Validator method): {speed_score_validator:.4f}")
    print(f"Speed Score (FPS method): {speed_score_fps:.2f}/100")
    print("-"*60)
    print(f"BBox Score: {bbox_score:.4f}")
    print(f"Keypoint Score: {keypoint_final_score:.4f}")
    print(f"Final Score: {final_score:.4f}")
    print("-"*60)
    print(f"Frame scores: {result.frame_scores}")
    print(f"Feedback: {result.feedback}")
    print("="*60)
    
    # Save detailed results to file
    results_summary = {
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
    
    with open("scoring_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: scoring_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 
