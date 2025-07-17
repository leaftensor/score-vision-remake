import cv2
import json
import requests
import numpy as np
import os
from pathlib import Path
from validator.evaluation.bbox_clip import data_processor as validator_data_processor, clip_model as validator_clip_model, BoundingBoxObject, OBJECT_ID_TO_ENUM
import torch

# --- Config ---
TEST_VIDEO_URL = "https://scoredata.me/2025_06_18/2025_06_18_d49f45ff/2025_06_18_d49f45ff_195945e20e9e4325b51ab84ff134c7_dcb7b85f.mp4"
RESULT_JSON_PATH = "C:/Users/longp/Documents/GitHub/score-vision/miner/test_outputs/pipeline_results_1752724503.json"
VIDEO_LOCAL_PATH = "miner/test_outputs/temp_test_video.mp4"

# --- Helper: Download video if not exists ---
def download_video(url, save_path):
    # if os.path.exists(save_path):
    #     print(f"Video already exists at {save_path}")
    #     return save_path
    print(f"Downloading video from {url} ...")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {save_path}")
    return save_path

def get_full_label_name(label):
    # Accepts either int (class_id), string, or validator enum string
    mapping = {
        0: "football",
        1: "goalkeeper",
        2: "player",
        3: "referee",
        "football": "football",
        "goalkeeper": "goalkeeper",
        "player": "player",
        "football player": "player",
        "referee": "referee",
        "crowd": "crowd",
        "grass": "grass",
        "goal": "goal",
        "background": "background",
        "blank": "blank",
        "other": "other",
        "not a football": "not a football",
        "black shape": "black shape",
        "BLACK": "black shape",
        "CROWD": "crowd",
        "GRASS": "grass",
        "GOAL": "goal",
        "BACKGROUND": "background",
        "BLANK": "blank",
        "OTHER": "other",
        "NOTFOOT": "not a football",
    }
    if hasattr(label, 'value'):
        label = label.value
    if isinstance(label, int):
        return mapping.get(label, f"class_{label}")
    if isinstance(label, str):
        return mapping.get(label, label)
    return str(label)

# --- Helper: Draw bboxes ---
def draw_bboxes(frame, objects, scores=None, expected_labels=None):
    for idx, obj in enumerate(objects):
        bbox = obj.get("bbox", [])
        class_id = obj.get("class_id", -1)
        color = (0, 255, 0)  # Default: green
        if class_id == 0:
            color = (0, 255, 255)  # Ball: yellow
        elif class_id == 1:
            color = (0, 0, 255)    # Goalkeeper: red
        elif class_id == 2:
            color = (0, 255, 0)    # Player: green
        elif class_id == 3:
            color = (255, 0, 0)    # Referee: blue
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Predicted/expected labels
            pred_label = obj.get('predicted_label')
            # Use provided expected_labels if available
            if expected_labels is not None and idx < len(expected_labels):
                exp_label = expected_labels[idx]
            else:
                exp_label = obj.get('expected_label')
            # Predicted: prefer predicted_label, fallback to class_id
            pred_name = get_full_label_name(pred_label) if pred_label is not None else get_full_label_name(class_id)
            exp_name = get_full_label_name(exp_label) if exp_label is not None else "?"
            label = f"ID:{obj.get('id','?')} Miner:{pred_name} Validator:{exp_name}"
            # Nếu có scores, overlay điểm lên bbox
            if scores is not None and idx < len(scores):
                label += f" S:{scores[idx]:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

# --- Helper: Parse keypoint scores from log ---
def parse_keypoint_scores(log_path):
    """
    Parse log file to extract per-frame scores from lines like:
    Frame 749:
    ...
    -> (Scaled by factor 1.00) = 0.47
    Returns: dict {frame_idx: score}
    """
    scores = {}
    current_frame = None
    # Try multiple encodings for robustness
    try:
        with open(log_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(log_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('Frame '):
            try:
                current_frame = int(line.split()[1].replace(':', ''))
                print(line)
            except Exception:
                current_frame = None
        elif line.startswith('-> (Scaled by factor') and current_frame is not None:
            try:
                score = float(line.split('=')[-1].strip())
                scores[current_frame] = score
            except Exception:
                continue
    print(scores)
    return scores

def get_expected_label_clip(frame, bbox):
    # frame: BGR numpy array
    # bbox: dict with 'bbox' and 'class_id'
    import numpy as np
    SCALE_FOR_CLIP = 4.0
    x1, y1, x2, y2 = map(int, bbox['bbox'])
    roi = frame[y1:y2, x1:x2, ::-1]  # BGR->RGB
    class_id = bbox.get('class_id', -1)
    def move_to_device(inputs, device):
        for k in inputs:
            if hasattr(inputs[k], 'to'):
                inputs[k] = inputs[k].to(device)
        return inputs
    # Football logic
    if class_id == 0:
        # Step 1: roundness
        round_inputs = validator_data_processor(
            text=["a photo of a round object on grass", "a random object"],
            images=[roi],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        round_inputs = move_to_device(round_inputs, validator_clip_model.device)
        with torch.no_grad():
            round_outputs = validator_clip_model(**round_inputs)
            round_probs = round_outputs.logits_per_image.softmax(dim=1)[0]
        if round_probs[0].item() < 0.5:
            return "not a football"
        # Step 2: semantic
        step2_inputs = validator_data_processor(
            text=[
                "a small soccer ball on the field",
                "just grass without any ball",
                "other"
            ],
            images=[roi],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        step2_inputs = move_to_device(step2_inputs, validator_clip_model.device)
        with torch.no_grad():
            step2_outputs = validator_clip_model(**step2_inputs)
            step2_probs = step2_outputs.logits_per_image.softmax(dim=1)[0]
        pred_idx = int(torch.argmax(step2_probs).item())
        if pred_idx == 0:
            return "football"
        else:
            return "not a football"
    # Person/grass logic
    step1_inputs = validator_data_processor(
        text=["person", "grass"],
        images=[roi],
        return_tensors="pt",
        padding=True
    )
    step1_inputs = move_to_device(step1_inputs, validator_clip_model.device)
    with torch.no_grad():
        step1_outputs = validator_clip_model(**step1_inputs)
        step1_probs = step1_outputs.logits_per_image.softmax(dim=1)[0]
    person_score = step1_probs[0].item()
    grass_score = step1_probs[1].item()
    if person_score > 0.08:
        # Refine
        person_labels = [
            "football player",
            "goalkeeper",
            "referee",
            "crowd",
            "black shape"
        ]
        refine_inputs = validator_data_processor(
            text=person_labels,
            images=[roi],
            return_tensors="pt",
            padding=True
        )
        refine_inputs = move_to_device(refine_inputs, validator_clip_model.device)
        with torch.no_grad():
            refine_outputs = validator_clip_model(**refine_inputs)
            refine_probs = refine_outputs.logits_per_image.softmax(dim=1)[0]
            refine_pred = int(torch.argmax(refine_probs).item())
        return person_labels[refine_pred]
    else:
        return "grass"

# --- Demo usage: print expected label for each object in viz_frame ---
def viz_frame(frame_idx):
    # 1. Download video if needed
    download_video(TEST_VIDEO_URL, VIDEO_LOCAL_PATH)
    # 2. Load result json
    with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames = data["frames"] if "frames" in data else data
    frame_data = None
    if str(frame_idx) in frames:
        frame_data = frames[str(frame_idx)]
    elif int(frame_idx) in frames:
        frame_data = frames[int(frame_idx)]
    else:
        print(f"Frame {frame_idx} not found in result json!")
        return
    # 3. Parse keypoint scores from log
    log_path = "C://Users/longp/Documents/GitHub/score-vision/output_complete_pipline.log"
    keypoint_scores = parse_keypoint_scores(log_path)
    score_str = None
    if frame_idx in keypoint_scores:
        score_str = f"Keypoint score: {keypoint_scores[frame_idx]:.2f}"
        print(score_str)
    else:
        print(f"Frame {frame_idx} not found in keypoint scores!")
    # --- Hard-code object scores for frame 749 ---
    hard_scores_749 = [0.50, 0.25, 1.00, 0.12, 0.50, 0.06, 0.25, 0.03, 0.12, 0.02, -1.00, 0.01, 0.00, 0.00]
    if frame_idx == 749:
        print(f"Hard-coded object scores for frame 749: {hard_scores_749}")
        # Overlay summary on image
        hard_score_str = f"Obj scores: {', '.join([str(x) for x in hard_scores_749])}"
    else:
        hard_score_str = None
    # 4. Open video, seek to frame_idx
    cap = cv2.VideoCapture(VIDEO_LOCAL_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read frame {frame_idx} from video!")
        return
    # 5. Draw bboxes
    objects = frame_data.get("objects", [])
    # Compute expected labels using CLIP
    expected_labels = []
    for obj in objects:
        try:
            exp_label = get_expected_label_clip(frame, obj)
        except Exception as e:
            exp_label = f"ERROR: {e}"
        expected_labels.append(exp_label)
    # Print expected label for each object using CLIP
    print("\nExpected label by CLIP for each object:")
    for obj, exp_label in zip(objects, expected_labels):
        print(f"Object ID {obj.get('id','?')}: {exp_label}")
    if frame_idx == 749:
        frame_viz = draw_bboxes(frame, objects, scores=hard_scores_749, expected_labels=expected_labels)
    else:
        frame_viz = draw_bboxes(frame, objects, expected_labels=expected_labels)
    # 6. Draw keypoint score if available
    if score_str:
        cv2.putText(frame_viz, score_str, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    # 6b. Draw hard-coded object scores for frame 749
    if hard_score_str:
        cv2.putText(frame_viz, hard_score_str, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    # 7. Show
    cv2.imshow(f"Frame {frame_idx}", frame_viz)
    print(f"Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize frame from result json and video")
    parser.add_argument("--frame", type=int, required=True, help="Frame index to visualize")
    args = parser.parse_args()
    viz_frame(args.frame) 