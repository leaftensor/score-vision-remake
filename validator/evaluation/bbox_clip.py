from argparse import ArgumentParser
from pathlib import Path
from json import load
from random import sample
from enum import Enum
from asyncio import get_event_loop, run
from logging import getLogger, basicConfig, INFO, DEBUG
from math import cos, acos, exp
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
from time import time

from numpy import ndarray, zeros
from pydantic import BaseModel, Field
from transformers import CLIPProcessor, CLIPModel
from cv2 import VideoCapture
from torch import no_grad
import torch
from typing import List, Dict

# Import TensorRT optimization
try:
    from .tensorrt_clip import create_tensorrt_clip
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

SCALE_FOR_CLIP = 4.0
FRAMES_PER_VIDEO = 750
MIN_WIDTH = 15
MIN_HEIGHT = 40
logger = getLogger("Bounding Box Evaluation Pipeline")
# Chuyển model CLIP và processor sang GPU nếu có
clip_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
data_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Global text feature cache
_text_feature_cache = {}

# CLIP result cache by tracking ID (for tracked objects)
_clip_tracking_cache = {}

# TensorRT disabled for stability - using PyTorch only

# Disable TensorRT due to CUDA context corruption issues
tensorrt_clip = None
print("TensorRT disabled due to CUDA stability issues, using optimized PyTorch CLIP")

# Set precision for better performance
torch.set_float32_matmul_precision('high')
print("Using optimized CLIP model with TensorRT acceleration" if tensorrt_clip else "Using standard PyTorch CLIP model")

class BoundingBoxObject(Enum):
    #possible classifications identified by miners and CLIP
    FOOTBALL = "football"
    GOALKEEPER = "goalkeeper"
    PLAYER = "football player"
    REFEREE = "referee"
    #possible additional classifications identified by CLIP only
    CROWD = "crowd"
    GRASS = "grass"
    GOAL = "goal"
    BACKGROUND = "background"
    BLANK = "blank"
    OTHER = "other"
    NOTFOOT="not a football"
    BLACK="black shape"

OBJECT_ID_TO_ENUM = {
    0:BoundingBoxObject.FOOTBALL,
    1:BoundingBoxObject.GOALKEEPER,
    2:BoundingBoxObject.PLAYER,
    3:BoundingBoxObject.REFEREE,
}


class BBox(BaseModel):
    x1:int
    y1:int
    x2:int
    y2:int

    @property
    def width(self) -> int:
        return abs(self.x2-self.x1)

    @property
    def height(self) -> int:
        return abs(self.y2-self.y1)

class BBoxScore(BaseModel):
    predicted_label:BoundingBoxObject = Field(..., description="Object type classified by the Miner's Model")
    expected_label:BoundingBoxObject = Field(..., description="Object type classified by the Validator's Model (CLIP)")
    occurrence:int = Field(..., description="The number of times an of object of this type has been seen up to now")

    def __str__(self) -> str:
        return f"""
expected: {self.expected_label.value}
predicted: {self.predicted_label.value}
    correctness: {self.correctness}
    validity: {self.validity}
        score: {self.score}
        weight: {self.weight}
            points = {self.points}
"""

    @property
    def validity(self) -> bool:
        """Is the Object captured by the Miner
        a valid object of interest (e.g. player, goalkeeper, football, ref)
        or is it another object we don't care about?"""
        return self.expected_label in OBJECT_ID_TO_ENUM.values()

    @property
    def correctness(self) -> bool:
        """"Does the Object type classified by the Miner's Model
        match the classification given by the Validator?"""
        return self.predicted_label==self.expected_label

    @property
    def weight(self) -> float:
        """The first time an object of a certain type is seen the weight is 1.0
        Thereafter it decreases exponentially.
        The infinite sum of 1/(2**n) converges to 2
        Which is useful to know for normalising total scores"""
        return 1/(2**self.occurrence)

    @property
    def score(self) -> float:
        """
        New :
        - GRASS / CROWD / OTHER / BACKGROUND → -1.0
        - PERSON (player, ref, goalkeeper)
            - correct → 1.0
            - wrong classification → 0.5
            - other → -1.0
        - FOOTBALL :
            - correct → 1.0
            - else → -1.0
        """
        if self.expected_label in {
            BoundingBoxObject.GRASS,
            BoundingBoxObject.CROWD,
            BoundingBoxObject.OTHER,
            BoundingBoxObject.BACKGROUND,
        }:
            return -1.0

        if self.expected_label in {
            BoundingBoxObject.PLAYER,
            BoundingBoxObject.GOALKEEPER,
            BoundingBoxObject.REFEREE,
        }:
            if self.predicted_label == self.expected_label:
                return 1.0
            elif self.predicted_label in {
                BoundingBoxObject.PLAYER,
                BoundingBoxObject.GOALKEEPER,
                BoundingBoxObject.REFEREE,
            }:
                return 0.5
            else:
                return -1.0

        if self.expected_label == BoundingBoxObject.FOOTBALL:
            if self.predicted_label == BoundingBoxObject.FOOTBALL:
                return 1.0
            else:
                return -1.0

        return -1.0


    @property
    def points(self) -> float:
        return self.weight*self.score



async def stream_frames(video_path:Path):
    cap = VideoCapture(str(video_path))
    try:
        frame_count = 0
        while True:
            ret, frame = await get_event_loop().run_in_executor(None, cap.read)
            if not ret:
                break
            yield frame_count, frame
            frame_count += 1
    finally:
        cap.release()

   

def multiplication_factor(image_array:ndarray, bboxes:list[dict[str,int|tuple[int,int,int,int]]]) -> float:
    """Reward more targeted bbox predictions
    while penalising excessively large or numerous bbox predictions
    """
    total_area_bboxes = 0.0
    valid_bbox_count = 0
    for bbox in bboxes:
        if 'bbox' not in bbox:
            continue
        x1,y1,x2,y2 = bbox['bbox']
        w = abs(x2-x1)
        h = abs(y2-y1)
        a = w*h
        total_area_bboxes += a
        valid_bbox_count += 1
    if valid_bbox_count==0:
        return 1.0
    height,width,_ = image_array.shape
    area_image = height*width
    logger.debug(f"Total BBox Area: {total_area_bboxes:.2f} pxl^2\nImage Area: {area_image:.2f} pxl^2")

    percentage_image_area_covered = total_area_bboxes / area_image
    avg_percentage_area_per_bbox=percentage_image_area_covered/valid_bbox_count

    if avg_percentage_area_per_bbox <= 0.015:
        if percentage_image_area_covered <= 0.15:
            scaling_factor=1
        else:
            scaling_factor = exp(-3 * (percentage_image_area_covered - 0.15))

    else:
        scaling_factor = exp(-80 * (avg_percentage_area_per_bbox - 0.015))

    logger.info(
        f"Avg area per object: {avg_percentage_area_per_bbox*100:.2f}% of image — scaling factor = {scaling_factor:.4f}"
    )
    return scaling_factor

def batch_classify_rois(regions_of_interest:list[ndarray]) -> list[BoundingBoxObject]:
    """Use CLIP to classify a batch of images (on GPU if available)"""
    model_inputs = data_processor(
        text=[key.value for key in BoundingBoxObject],
        images=regions_of_interest,
        return_tensors="pt",
        padding=True
    ).to(clip_device)
    with torch.no_grad():
        model_outputs = clip_model(**model_inputs)
        probabilities = model_outputs.logits_per_image.softmax(dim=1)
        object_ids = probabilities.argmax(dim=1)
    return [OBJECT_ID_TO_ENUM.get(object_id.item(), BoundingBoxObject.OTHER) for object_id in object_ids]

def crop_and_return_scaled_roi(image: ndarray, bbox: tuple[int, int, int, int], scale: float) -> ndarray | None:
    x1, y1, x2, y2 = map(float, bbox)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or scale < 1.0:
        return None

    cx, cy = x1 + w / 2, y1 + h / 2
    half_w, half_h = (w * scale) / 2.0, (h * scale) / 2.0

    nx1, ny1 = cx - half_w, cy - half_h
    nx2, ny2 = cx + half_w, cy + half_h

    H, W = image.shape[:2]
    nx1, nx2 = max(0, int(round(nx1))), min(W - 1, int(round(nx2)))
    ny1, ny2 = max(0, int(round(ny1))), min(H - 1, int(round(ny2)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return image[ny1:ny2, nx1:nx2].copy()

def extract_regions_of_interest_from_image(bboxes:list[dict], image_array:ndarray) -> list[ndarray]:
    rois = []
    for bbox in bboxes:
        coords = bbox['bbox']
        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])
            roi = image_array[y1:y2, x1:x2, :].copy() 
            rois.append(roi)
            image_array[y1:y2, x1:x2, :] = 0  
    return rois
  
def is_bbox_large_enough(bbox_dict):
    coords = bbox_dict["bbox"]
    if not (isinstance(coords, (list, tuple)) and len(coords) == 4):
        return False
    x1, y1, x2, y2 = coords
    w, h = x2 - x1, y2 - y1
    class_id = bbox_dict.get("class_id", -1)
    if class_id == 0:  # FOOTBALL
        return True
    return w >= MIN_WIDTH and h >= MIN_HEIGHT

def is_touching_scoreboard_zone(bbox_dict, frame_width=1280, frame_height=720):
    x1, y1, x2, y2 = bbox_dict["bbox"]
    
    scoreboard_top = 0
    scoreboard_bottom = 150
    scoreboard_left = 0
    scoreboard_right = frame_width

    # If the bbox intersects with the top area, it's out of bounds
    intersects_top = not (x2 < scoreboard_left or x1 > scoreboard_right or y2 < scoreboard_top or y1 > scoreboard_bottom)
    return intersects_top
    
def evaluate_frame(
    frame_id:int,
    image_array:ndarray,
    bboxes:list[dict[str,int|tuple[int,int,int,int]]]
) -> float:
    object_counts = {
        BoundingBoxObject.FOOTBALL: 0,
        BoundingBoxObject.GOALKEEPER: 0,
        BoundingBoxObject.PLAYER: 0,
        BoundingBoxObject.REFEREE: 0,
        BoundingBoxObject.OTHER: 0
    }
    bboxes = [bbox for bbox in bboxes if is_bbox_large_enough(bbox) and not is_touching_scoreboard_zone(bbox, image_array.shape[1], image_array.shape[0])]
    if not bboxes:
        logger.info(f"Frame {frame_id}: all bboxes filtered out due to small size or corners — skipping.")
        return 0.0
        
    rois = extract_regions_of_interest_from_image(
        bboxes=bboxes,
        image_array=image_array[:,:,::-1] # BGR -> RGB
    )

    predicted_labels = [
        OBJECT_ID_TO_ENUM.get(bbox["class_id"], BoundingBoxObject.OTHER)
        for bbox in bboxes
    ]

    # Step 1 : "person" vs "grass"
    step1_inputs = data_processor(
        text=["person", "grass"],
        images=rois,
        return_tensors="pt",
        padding=True
    ).to(clip_device)
    with torch.no_grad():
        step1_outputs = clip_model(**step1_inputs)
        step1_probs = step1_outputs.logits_per_image.softmax(dim=1)

    expected_labels: list[BoundingBoxObject] = [None] * len(rois)
    rois_for_person_refine = []
    indexes_for_person_refine = []

    # Football or other
    ball_indexes = [i for i, pred in enumerate(predicted_labels) if pred == BoundingBoxObject.FOOTBALL]
    person_candidate_indexes = [i for i in range(len(rois)) if i not in ball_indexes]

    # Step 2a : FOOTBALL with new 2-step CLIP check
    if ball_indexes:
        if len(ball_indexes)>1:
            logger.info(f"Frame {frame_id} has {len(ball_indexes)} footballs — keeping only the first.")
            for i in ball_indexes[1:]:
                predicted_labels[i] = BoundingBoxObject.FOOTBALL  # Still count as predicted
                expected_labels[i] = BoundingBoxObject.NOTFOOT     # Marked as incorrect
            ball_indexes = [ball_indexes[0]]
        
        football_rois = [
            crop_and_return_scaled_roi(image_array[:,:,::-1], bboxes[i]['bbox'], scale=SCALE_FOR_CLIP)
            for i in ball_indexes
        ]
        football_rois = [roi for roi in football_rois if roi is not None]

        # Step 1: round object
        round_inputs = data_processor(
            text=["a photo of a round object on grass", "a random object"],
            images=football_rois,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(clip_device)

        with torch.no_grad():
            round_outputs = clip_model(**round_inputs)
            round_probs = round_outputs.logits_per_image.softmax(dim=1)

        # Step 2: semantic football if round enough
        for j, i in enumerate(ball_indexes):
            round_prob = round_probs[j][0].item()

            if round_prob < 0.5:
                expected_labels[i] = BoundingBoxObject.NOTFOOT
                continue

            # Step 2 — semantic football classification
            step2_inputs = data_processor(
                text=[
                "a small soccer ball on the field",
                "just grass without any ball",
                "other"
                ],
                images=[football_rois[j]],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(clip_device)

            with torch.no_grad():
                step2_outputs = clip_model(**step2_inputs)
                step2_probs = step2_outputs.logits_per_image.softmax(dim=1)[0]

            pred_idx = torch.argmax(step2_probs).item()
            if pred_idx == 0:  # ball or close-up of ball
                expected_labels[i] = BoundingBoxObject.FOOTBALL
            else:
                expected_labels[i] = BoundingBoxObject.NOTFOOT

    # step 2b : PERSON vs GRASS
    for i in person_candidate_indexes:
        person_score = step1_probs[i][0].item()
        grass_score = step1_probs[i][1].item()
        if person_score > 0.08:
            rois_for_person_refine.append(rois[i])
            indexes_for_person_refine.append(i)
        else:
            expected_labels[i] = BoundingBoxObject.GRASS

    # Step 3 : PERSON final classification
    if rois_for_person_refine:
        person_labels = [
            BoundingBoxObject.PLAYER.value,
            BoundingBoxObject.GOALKEEPER.value,
            BoundingBoxObject.REFEREE.value,
            BoundingBoxObject.CROWD.value,
            BoundingBoxObject.BLACK.value
        ]
        refine_inputs = data_processor(
            text=person_labels,
            images=rois_for_person_refine,
            return_tensors="pt",
            padding=True
        ).to(clip_device)
        with torch.no_grad():
            refine_outputs = clip_model(**refine_inputs)
            refine_probs = refine_outputs.logits_per_image.softmax(dim=1)
            refine_preds = refine_probs.argmax(dim=1)

        for k, idx in enumerate(indexes_for_person_refine):
            expected_labels[idx] = BoundingBoxObject(person_labels[refine_preds[k]])

    # Scoring
    scores = []
    for i in range(len(expected_labels)):
        if expected_labels[i] is None:
            continue
        predicted = predicted_labels[i]
        expected = expected_labels[i]
        # print(predicted, expected)
        scores.append(
            BBoxScore(
                predicted_label=predicted,
                expected_label=expected,
                occurrence=len([
                    s for s in scores if s.expected_label == expected
                ])
            )
        )

    logger.debug('\n'.join(map(str, scores)))
    points = [score.points for score in scores]
    total_points = sum(points)
    n_unique_classes_detected = len(set(expected_labels) - {None})
    normalised_score = total_points / (n_unique_classes_detected+1) if n_unique_classes_detected else 0.0
    scale = multiplication_factor(image_array=image_array, bboxes=bboxes)
    scaled_score = scale * normalised_score

    points_with_labels = [f"{score.points:.2f} (predicted:{score.predicted_label.value})(expected:{score.expected_label.value})" for score in scores]
    logger.info(
        f"Frame {frame_id}:\n"
        f"\t-> {len(bboxes)} Bboxes predicted\n"
        f"\t-> sum({', '.join(points_with_labels)}) = {total_points:.2f}\n"
        f"\t-> (normalised by {n_unique_classes_detected} classes detected) = {normalised_score:.2f}\n"
        f"\t-> (Scaled by factor {scale:.2f}) = {scaled_score:.2f}"
    )
    print(
        f"Frame {frame_id}:\n"
        f"\t-> {len(bboxes)} Bboxes predicted\n"
        f"\t-> sum({', '.join(points_with_labels)}) = {total_points:.2f}\n"
        f"\t-> (normalised by {n_unique_classes_detected} classes detected) = {normalised_score:.2f}\n"
        f"\t-> (Scaled by factor {scale:.2f}) = {scaled_score:.2f}"
    )
    return scaled_score

# Mapping from expected_label to class_id
EXPECTED_LABEL_TO_CLASS_ID = {
    BoundingBoxObject.FOOTBALL: 0,
    BoundingBoxObject.GOALKEEPER: 1,
    BoundingBoxObject.PLAYER: 2,
    BoundingBoxObject.REFEREE: 3,
    BoundingBoxObject.CROWD: -1,
    BoundingBoxObject.GRASS: -1,
    BoundingBoxObject.GOAL: -1,
    BoundingBoxObject.BACKGROUND: -1,
    BoundingBoxObject.BLANK: -1,
    BoundingBoxObject.OTHER: -1,
    BoundingBoxObject.NOTFOOT: -1,
    BoundingBoxObject.BLACK: -1,
}

async def evaluate_bboxes(prediction:dict, path_video:Path, n_frames:int, n_valid:int) -> float:
    frames = prediction
    
    # Skip evaluation if no frames or all frames are empty
    if not frames or all(not frame.get("objects") for frame in frames.values()):
        logger.warning("No valid frames with objects found in prediction — skipping evaluation.")
        return 0.0
        
    if isinstance(frames, list):
        logger.warning("Legacy formatting detected. Updating...")
        frames = {
            frame.get('frame_number',str(i)):frame
            for i, frame in enumerate(frames)
        }

    frames_ids_which_can_be_validated = [
        frame_id for frame_id,predictions in frames.items()
        if any(predictions.get('objects',[]))
    ]
    frame_ids_to_evaluate=sample(
        frames_ids_which_can_be_validated,
        k=min(n_frames,len(frames_ids_which_can_be_validated))
    )

    if len(frame_ids_to_evaluate)/n_valid<0.7:
        logger.warning(f"Only having {len(frame_ids_to_evaluate)} which is not enough for the threshold")
        return 0.0
        
    if not any(frame_ids_to_evaluate):
        logger.warning("""
            We are currently unable to validate frames with no bboxes predictions
            It may be correct that there are no objects of interest within a frame
            and so we cannot simply give a score of 0.0 for no bbox predictions
            However, with our current method, we cannot confirm nor deny this
            So any frames without any predictions are skipped in favour of those which can be verified

            However, after skipping such frames (i.e. without bbox predictions),
            there were insufficient frames remaining upon which to base an accurate evaluation
            and so were forced to return a final score of 0.0
        """)
        return 0.0

    n_threads = min(cpu_count(),len(frame_ids_to_evaluate))
    logger.info(f"Loading Video: {path_video} to evaluate {len(frame_ids_to_evaluate)} frames (using {n_threads} threads)...")
    scores = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        async for frame_id,image in stream_frames(video_path=path_video):
            if str(frame_id) not in frame_ids_to_evaluate:
                continue
            futures.append(
                executor.submit(
                    evaluate_frame,frame_id,image,frames[str(frame_id)]['objects']
                )
            )
    for future in as_completed(futures):
        try: 
            score = future.result()
            scores.append(score)
        except Exception as e:
            logger.warning(f"Error while getting score from future: {e}")

    average_score = sum(scores)/len(scores) if scores else 0.0
    logger.info(f"Average Score: {average_score:.2f} when evaluated on {len(scores)} frames")
    return max(0.0,min(1.0,round(average_score,2)))


import numpy as np
import time

def get_cached_text_features(labels_tuple):
    """Get cached text features or compute and cache them"""
    global _text_feature_cache, tensorrt_clip
    
    if labels_tuple in _text_feature_cache:
        return _text_feature_cache[labels_tuple]
    
    # Use TensorRT if available, otherwise fallback to PyTorch
    if tensorrt_clip is not None:
        text_features = tensorrt_clip.get_cached_text_features(labels_tuple)
    else:
        # Compute text features with PyTorch
        text_inputs = data_processor(text=list(labels_tuple), return_tensors="pt", padding=True).to(clip_device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        del text_inputs
    
    # Cache the result
    _text_feature_cache[labels_tuple] = text_features
    return text_features

def calculate_bbox_importance(bbox, w, h, img_width, img_height, class_id):
    """Calculate importance score for bbox selection - higher score = better quality objects"""
    score = 0.0
    
    # 1. Class priority (footballs are most important for scoring)
    if class_id == 0:  # FOOTBALL
        score += 1000.0  # Extremely high priority for footballs
    elif class_id in [1, 2, 3]:  # GOALKEEPER, PLAYER, REFEREE
        score += 500.0   # Very high priority for people
    else:
        score += 1.0     # Much lower priority for unknown objects
    
    # 2. Enhanced size scoring - prioritize optimal sizes for each class
    area = w * h
    img_area = img_width * img_height
    relative_area = area / img_area
    
    if class_id == 0:  # Football - smaller optimal size
        if 0.0005 <= relative_area <= 0.01:  # Optimal football size
            score += 50.0
        elif 0.0002 <= relative_area <= 0.02:  # Good football size
            score += 30.0
        else:
            score += 5.0
    elif class_id in [1, 2, 3]:  # People - larger optimal size
        if 0.005 <= relative_area <= 0.08:  # Optimal people size
            score += 50.0
        elif 0.002 <= relative_area <= 0.15:  # Good people size
            score += 30.0
        else:
            score += 10.0
    else:  # Other objects
        if 0.001 <= relative_area <= 0.05:
            score += 20.0
        else:
            score += 5.0
    
    # 3. Enhanced position scoring - prioritize action areas
    center_x = (bbox["bbox"][0] + bbox["bbox"][2]) / 2
    center_y = (bbox["bbox"][1] + bbox["bbox"][3]) / 2
    
    # Distance from center (normalized)
    dx = abs(center_x - img_width/2) / (img_width/2)
    dy = abs(center_y - img_height/2) / (img_height/2)
    center_distance = (dx + dy) / 2
    
    # Closer to center = higher score (main action area)
    position_score = (1.0 - center_distance) * 30.0
    score += position_score
    
    # 4. Enhanced aspect ratio scoring - class-specific optimal ratios
    aspect_ratio = w / h if h > 0 else 1.0
    
    if class_id == 0:  # Football should be roughly square
        if 0.8 <= aspect_ratio <= 1.2:  # Very good football shape
            score += 25.0
        elif 0.6 <= aspect_ratio <= 1.6:  # Good football shape
            score += 15.0
        else:
            score += 5.0
    elif class_id in [1, 2, 3]:  # People should be taller than wide
        if 0.35 <= aspect_ratio <= 0.65:  # Very good people shape
            score += 25.0
        elif 0.25 <= aspect_ratio <= 0.85:  # Good people shape
            score += 15.0
        else:
            score += 8.0
    
    # 5. Tracking consistency bonus (tracked objects are more reliable)
    if "id" in bbox and bbox["id"] is not None:
        score += 100.0  # Higher bonus for tracked objects (continuity)
    
    # 6. Object completeness scoring - prefer objects not at image edges
    x1, y1, x2, y2 = bbox["bbox"]
    edge_margin = 20  # pixels
    
    # Check if object is cut off at edges
    if (x1 <= edge_margin or x2 >= img_width - edge_margin or 
        y1 <= edge_margin or y2 >= img_height - edge_margin):
        score -= 20.0  # Penalty for edge objects (likely cut off)
    else:
        score += 10.0  # Bonus for complete objects
    
    # 7. Enhanced quality penalties for poor detections
    if area < 100:  # Very tiny objects
        score -= 30.0
    if aspect_ratio > 4.0 or aspect_ratio < 0.15:  # Extremely distorted objects
        score -= 40.0
    
    # 8. Detection confidence bonus (if available)
    if "confidence" in bbox and bbox["confidence"] is not None:
        confidence_bonus = float(bbox["confidence"]) * 20.0
        score += confidence_bonus
    
    return max(10.0, score)  # Ensure minimum score to avoid filtering too aggressively

def hierarchical_classification_pipeline(roi_map, all_rois, images):
    """
    Hierarchical Intelligence Pipeline - Multi-stage classification
    Returns expected_labels with hierarchical filtering results
    """
    import time
    import numpy as np
    
    print(f"[HIERARCHICAL] Starting multi-stage classification pipeline for {len(all_rois)} ROIs")
    
    expected_labels = []
    stage1_filtered = 0
    stage2_filtered = 0
    
    # STAGE 1: Ultra-Fast Heuristics (0.1ms per ROI)
    t_stage1_start = time.time()
    stage1_results = []
    
    for i, roi in enumerate(all_rois):
        frame_idx, obj_idx, bbox = roi_map[i]
        predicted_class = bbox.get("class_id", -1)
        predicted_label = OBJECT_ID_TO_ENUM.get(predicted_class, BoundingBoxObject.OTHER)
        
        # Get basic ROI properties
        h, w = roi.shape[:2]
        area = h * w
        
        # Get frame dimensions and bbox coordinates
        frame_height, frame_width = images[frame_idx].shape[:2]
        bbox_coords = bbox.get("bbox", [0, 0, 0, 0])
        
        stage1_confident = False
        
        # AGGRESSIVE Ultra-fast spatial filter (targeting 30-40% filtering rate)
        if len(bbox_coords) == 4:
            x1, y1, x2, y2 = bbox_coords
            y_center = y1 + (y2 - y1) // 2
            x_center = x1 + (x2 - x1) // 2
            
            # Much more aggressive spatial filtering
            if predicted_label == BoundingBoxObject.PLAYER:
                # Top 20% = likely crowd (more aggressive)
                if y_center < frame_height * 0.20:
                    stage1_results.append(BoundingBoxObject.CROWD)
                    stage1_confident = True
                    stage1_filtered += 1
                # Bottom 10% = likely UI elements (more aggressive)
                elif y_center > frame_height * 0.90:
                    stage1_results.append(BoundingBoxObject.OTHER)
                    stage1_confident = True
                    stage1_filtered += 1
                # Left/Right edges = likely crowd/UI (new filter)
                elif x_center < frame_width * 0.05 or x_center > frame_width * 0.95:
                    stage1_results.append(BoundingBoxObject.CROWD)
                    stage1_confident = True
                    stage1_filtered += 1
        
        # AGGRESSIVE Ultra-fast size filter (targeting more filtering)
        if not stage1_confident:
            if area < 800:  # Increase threshold = more aggressive
                stage1_results.append(BoundingBoxObject.OTHER)
                stage1_confident = True
                stage1_filtered += 1
            elif area > 40000:  # Decrease threshold = more aggressive
                stage1_results.append(BoundingBoxObject.BACKGROUND)
                stage1_confident = True
                stage1_filtered += 1
        
        # AGGRESSIVE Aspect ratio filter (new aggressive filter)
        if not stage1_confident and h > 0 and w > 0:
            aspect_ratio = w / h
            if aspect_ratio > 5.0:  # Very wide = likely line/background
                stage1_results.append(BoundingBoxObject.BACKGROUND)
                stage1_confident = True
                stage1_filtered += 1
            elif aspect_ratio < 0.2:  # Very tall = likely line/noise
                stage1_results.append(BoundingBoxObject.OTHER)
                stage1_confident = True
                stage1_filtered += 1
        
        if stage1_confident:
            expected_labels.append(stage1_results[-1])
        else:
            stage1_results.append(None)
            expected_labels.append(None)  # Will be processed in stage 2
    
    t_stage1_end = time.time()
    print(f"[HIERARCHICAL] Stage 1 filtered: {stage1_filtered} ROIs in {(t_stage1_end - t_stage1_start)*1000:.1f}ms")
    
    # STAGE 2: Visual Feature Analysis (1ms per ROI)
    t_stage2_start = time.time()
    stage2_results = []
    
    for i, (stage1_result, roi) in enumerate(zip(stage1_results, all_rois)):
        if stage1_result is not None:
            stage2_results.append(stage1_result)
            continue
            
        frame_idx, obj_idx, bbox = roi_map[i]
        predicted_class = bbox.get("class_id", -1)
        predicted_label = OBJECT_ID_TO_ENUM.get(predicted_class, BoundingBoxObject.OTHER)
        
        # Get ROI properties
        h, w = roi.shape[:2]
        area = h * w
        
        stage2_confident = False
        
        # AGGRESSIVE Visual feature analysis (targeting 20-30% additional filtering)
        if roi.size > 0:
            mean_color = roi.mean(axis=(0,1))
            std_color = roi.std(axis=(0,1))
            
            # Much more aggressive color-based filtering
            if predicted_label == BoundingBoxObject.PLAYER:
                # Green = grass with lower threshold (more aggressive)
                green_dominance = mean_color[1] - max(mean_color[0], mean_color[2])
                if green_dominance > 20:  # Much lower threshold
                    stage2_results.append(BoundingBoxObject.GRASS)
                    stage2_confident = True
                    stage2_filtered += 1
                # Dark regions = shadow with higher threshold (more aggressive)
                elif mean_color.max() < 50:  # Higher threshold
                    stage2_results.append(BoundingBoxObject.OTHER)
                    stage2_confident = True
                    stage2_filtered += 1
                # Very bright regions = background
                elif mean_color.min() > 180:  # New filter
                    stage2_results.append(BoundingBoxObject.BACKGROUND)
                    stage2_confident = True
                    stage2_filtered += 1
                # Low color variance = likely uniform background
                elif std_color.mean() < 15:  # New filter
                    stage2_results.append(BoundingBoxObject.BACKGROUND)
                    stage2_confident = True
                    stage2_filtered += 1
        
        # AGGRESSIVE Size-based filtering for stage 2
        if not stage2_confident:
            if predicted_label == BoundingBoxObject.PLAYER:
                if area < 600:  # Increase threshold = more aggressive
                    stage2_results.append(BoundingBoxObject.OTHER)
                    stage2_confident = True
                    stage2_filtered += 1
                elif area > 35000:  # Decrease threshold = more aggressive
                    stage2_results.append(BoundingBoxObject.CROWD)
                    stage2_confident = True
                    stage2_filtered += 1
            # Apply size filters to all classes, not just PLAYER
            elif area < 300:  # Very small objects
                stage2_results.append(BoundingBoxObject.OTHER)
                stage2_confident = True
                stage2_filtered += 1
            elif area > 50000:  # Very large objects
                stage2_results.append(BoundingBoxObject.BACKGROUND)
                stage2_confident = True
                stage2_filtered += 1
        
        if stage2_confident:
            expected_labels[i] = stage2_results[-1]
        else:
            stage2_results.append(None)
            # Will be processed by CLIP in stage 3
    
    t_stage2_end = time.time()
    print(f"[HIERARCHICAL] Stage 2 filtered: {stage2_filtered} ROIs in {(t_stage2_end - t_stage2_start)*1000:.1f}ms")
    
    # Calculate how many ROIs need CLIP processing (Stage 3)
    clip_needed = sum(1 for label in expected_labels if label is None)
    total_hierarchical_filtered = stage1_filtered + stage2_filtered
    hierarchical_success_rate = (total_hierarchical_filtered / len(all_rois)) * 100 if all_rois else 0
    
    print(f"[HIERARCHICAL] Total filtered: {total_hierarchical_filtered}/{len(all_rois)} ROIs ({hierarchical_success_rate:.1f}%)")
    print(f"[HIERARCHICAL] CLIP processing needed: {clip_needed} ROIs")
    
    return expected_labels

def batch_evaluate_frame_filter(frames: List[Dict], images: List[np.ndarray], batch_size: int = None, enable_class_limits: bool = False):
    """
    Batch evaluate frame filter with optional class limiting.
    """
    import time
    import cv2
    global clip_device
    t0 = time.time()
    # --- ROI extraction ---
    t_roi_start = time.time()
    # Auto-determine optimal batch size based on GPU memory and available ROIs
    if batch_size is None:
        if clip_device.type == 'cuda':
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                if gpu_mem >= 32:  # A100 or similar
                    batch_size = 4096
                elif gpu_mem >= 16:  # RTX 4090, V100
                    batch_size = 2048
                elif gpu_mem >= 8:   # RTX 3080, RTX 4070
                    batch_size = 1024
                else:               # Lower-end GPUs
                    batch_size = 512
            except:
                batch_size = 1024  # Safe default
        else:
            batch_size = 256  # CPU fallback
    
    print(f"Using CLIP batch size: {batch_size} (GPU: {clip_device})")
    
    # 1. RELAXED ROI extraction - keep more objects
    all_rois = []
    roi_map = []  # (frame_idx, obj_idx, bbox_dict)
    total_bboxes = 0
    total_valid_bboxes = 0
    total_valid_rois = 0
    
    # Pre-convert all images to RGB in one batch
    rgb_images = []
    for img in images:
        rgb_images.append(img[:,:,::-1])  # BGR -> RGB batch conversion
    
    # Process frames with RELAXED filtering
    for frame_idx, frame_data in enumerate(frames):
        bboxes = frame_data.get("objects", [])
        if not bboxes:
            continue
        total_bboxes += len(bboxes)
        
        # Get image info once
        img_height, img_width = images[frame_idx].shape[:2]
        img_rgb = rgb_images[frame_idx]
        
        # RELAXED filtering for all bboxes - keep more objects
        valid_rois = []
        valid_bboxes = []
        bbox_scores = []
        
        for bbox in bboxes:
            coords = bbox["bbox"]
            if not (isinstance(coords, (list, tuple)) and len(coords) == 4):
                continue
            
            x1, y1, x2, y2 = coords
            w, h = x2 - x1, y2 - y1
            class_id = bbox.get("class_id", -1)
            
            # VERY RELAXED size filter - minimum requirements
            min_width = 8 if class_id == 0 else 10  # Reduced from 10/12
            min_height = 15 if class_id == 0 else 20  # Reduced from 20/25
            
            if not (w >= min_width and h >= min_height):
                continue
            
            # VERY RELAXED scoreboard filter - only top 80 pixels instead of 100
            if y1 < 80:  # Reduced from 100
                continue
            
            # VERY RELAXED edge filter - reduced margin from 30 to 20
            edge_margin = 20  # Reduced from 30
            if (
                x1 <= edge_margin or x2 >= img_width - edge_margin or 
                y1 <= edge_margin or y2 >= img_height - edge_margin
            ):
                continue
            
            # Bounds checking and ROI extraction
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_width, int(x2)), min(img_height, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
                
            roi = img_rgb[y1:y2, x1:x2].copy()
            if roi.size == 0:
                continue
                
            # Resize small ROIs for consistency
            if roi.shape[0] < 32 or roi.shape[1] < 32:
                roi = cv2.resize(roi, (32, 32))
            
            # Calculate importance score (more generous)
            importance_score = calculate_bbox_importance(bbox, w, h, img_width, img_height, class_id)
            
            valid_rois.append(roi)
            valid_bboxes.append(bbox)
            bbox_scores.append(importance_score)
        
        # RELAXED overlap removal - keep more overlapping objects
        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            return iou
        
        keep_indices = []
        for i, bbox_i in enumerate(valid_bboxes):
            boxA = bbox_i["bbox"]
            overlap = False
            for j in keep_indices:
                boxB = valid_bboxes[j]["bbox"]
                # VERY RELAXED IoU threshold from 0.6 to 0.8 - allow much more overlapping objects
                if compute_iou(boxA, boxB) > 0.8:  # Increased from 0.6
                    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                    boxB = valid_bboxes[j]["bbox"]
                    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                    if areaA > areaB:
                        keep_indices.remove(j)
                        break
                    else:
                        overlap = True
                        break
            if not overlap:
                keep_indices.append(i)
        
        valid_rois = [valid_rois[i] for i in keep_indices]
        valid_bboxes = [valid_bboxes[i] for i in keep_indices]
        bbox_scores = [bbox_scores[i] for i in keep_indices]

        # CLASS LIMITS FILTER REMOVED - Keep all valid objects without limits
        # The enable_class_limits filter has been completely removed to allow maximum objects
        
        total_valid_bboxes += len(valid_bboxes)
        
        # Add to global lists efficiently
        for obj_idx, (roi, bbox) in enumerate(zip(valid_rois, valid_bboxes)):
            all_rois.append(roi)
            roi_map.append((frame_idx, obj_idx, bbox))
            total_valid_rois += 1
            
    reduction_percent = (1.0 - total_valid_bboxes / total_bboxes) * 100 if total_bboxes > 0 else 0
    print(f"Total frames: {len(frames)} | Total bboxes: {total_bboxes} | Valid bboxes: {total_valid_bboxes} | Valid ROIs: {total_valid_rois}")
    
    if enable_class_limits:
        print(f"RELAXED 4-class selection: {reduction_percent:.1f}% | Max 12 best per class | Avg objects per frame: {total_valid_bboxes/len(frames):.1f}")
        t1 = time.time(); print(f"[TIMING] ROI gathering with relaxed 4-class selection: {t1-t0:.3f}s")
    else:
        print(f"RELAXED object processing: {reduction_percent:.1f}% | All classes | Avg objects per frame: {total_valid_bboxes/len(frames):.1f}")
        t1 = time.time(); print(f"[TIMING] ROI gathering with relaxed selection: {t1-t0:.3f}s")
    t_roi_end = time.time()
    print(f"[TIMING] ROI extraction: {t_roi_end - t_roi_start:.3f}s")
    # --- Heuristic filtering ---
    t_heur_start = time.time()
    # Get miner predictions for smart routing
    predicted_labels = [OBJECT_ID_TO_ENUM.get(bbox.get("class_id", -1), BoundingBoxObject.OTHER) for _, _, bbox in roi_map]
    # Initialize expected labels with heuristic defaults
    expected_labels = []
    # VERY RELAXED heuristic approach - keep most objects for CLIP
    expected_labels = []
    for i, (predicted_label, roi) in enumerate(zip(predicted_labels, all_rois)):
        h, w = roi.shape[:2]
        area = h * w
        
        # Check tracking cache first for tracked objects
        _, _, bbox = roi_map[i]
        tracking_id = bbox.get("id")
        if tracking_id is not None and tracking_id in _clip_tracking_cache:
            expected_labels.append(_clip_tracking_cache[tracking_id])
            continue
        
        # VERY MINIMAL heuristic filtering - only obvious non-objects
        if (predicted_label not in [0, 1, 2, 3] and  # Not key objects
            area < 50 and roi.size > 0):  # Very small areas only
            
            # Ultra-fast grass check - but more lenient
            mean_color = roi.mean(axis=(0,1))
            if (mean_color[1] > mean_color[0] + 35 and  # Very strong green
                mean_color[1] > mean_color[2] + 25):   # Very strong green dominance
                expected_labels.append(BoundingBoxObject.GRASS)
                continue
        
        # Send everything else to CLIP - keep more objects
        expected_labels.append(None)
    
    # Collect all cases that need CLIP processing
    clip_indices = []
    for i, expected in enumerate(expected_labels):
        if expected is None:
            clip_indices.append(i)
    
    # ... rest of CLIP processing remains the same ...
    # Process all uncertain cases with CLIP (but use optimized approach)
    if clip_indices:
        print(f"[CLIP] Processing {len(clip_indices)} uncertain ROIs out of {len(all_rois)}")
        
        # Use the exact same classification as the original for maximum accuracy
        all_labels = (
            BoundingBoxObject.PLAYER.value,      # 0
            BoundingBoxObject.GOALKEEPER.value,  # 1
            BoundingBoxObject.REFEREE.value,     # 2
            BoundingBoxObject.CROWD.value,       # 3
            BoundingBoxObject.BLACK.value,       # 4
            BoundingBoxObject.GRASS.value,       # 5
            BoundingBoxObject.FOOTBALL.value,    # 6
            BoundingBoxObject.NOTFOOT.value,     # 7
            BoundingBoxObject.BACKGROUND.value,  # 8
            BoundingBoxObject.OTHER.value        # 9
        )
        
        index_to_enum = {
            0: BoundingBoxObject.PLAYER,
            1: BoundingBoxObject.GOALKEEPER,
            2: BoundingBoxObject.REFEREE,
            3: BoundingBoxObject.CROWD,
            4: BoundingBoxObject.BLACK,
            5: BoundingBoxObject.GRASS,
            6: BoundingBoxObject.FOOTBALL,
            7: BoundingBoxObject.NOTFOOT,
            8: BoundingBoxObject.BACKGROUND,
            9: BoundingBoxObject.OTHER
        }
        
        # Use cached text features for massive speedup
        text_features = get_cached_text_features(all_labels)
        
        # Process uncertain ROIs with TensorRT optimization when available
        clip_rois = [all_rois[i] for i in clip_indices]
        # Optimize batch size for better GPU utilization and memory efficiency
        if batch_size is None:
            # Dynamic batch sizing based on ROI count and available memory
            if len(clip_rois) <= 8:
                clip_batch_size = len(clip_rois)  # Very small batches - process all at once
            elif len(clip_rois) <= 32:
                clip_batch_size = 8  # Small batches - good for memory efficiency
            elif len(clip_rois) <= 128:
                clip_batch_size = 16  # Medium batches - optimal for most GPUs
            elif len(clip_rois) <= 512:
                clip_batch_size = 32  # Large batches - balance memory and throughput
            else:
                clip_batch_size = 64  # Very large batches - maximize throughput
        else:
            clip_batch_size = min(batch_size, len(clip_rois))
        
        clip_predictions = []
        
        # Use PyTorch-only processing for stability
        if True:  # Always use PyTorch
            clip_batch_size = 512
            # PyTorch path - safe inference with CUDA error handling
            print(f"[PyTorch] Processing {len(clip_rois)} ROIs with stable PyTorch (batch_size={clip_batch_size})")
            
            # Try CUDA first, fall back to CPU if CUDA context is corrupted
            try:
                # Check if CUDA is available and working
                if clip_device.type == 'cuda':
                    # Test CUDA availability
                    torch.cuda.current_device()
                    
                with torch.amp.autocast('cuda') if clip_device.type == 'cuda' else torch.no_grad():
                    for i in range(0, len(clip_rois), clip_batch_size):
                        batch_rois = clip_rois[i:i+clip_batch_size]
                        
                        with torch.no_grad():
                            try:
                                # Try CUDA processing
                                image_inputs = data_processor(images=batch_rois, return_tensors="pt").to(clip_device)
                                image_features = clip_model.get_image_features(**image_inputs)
                                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                                
                                # Direct computation with confidence scoring
                                logits = torch.matmul(image_features, text_features.T) * clip_model.logit_scale.exp()
                                
                                # Use temperature scaling for better confidence calibration
                                probabilities = torch.softmax(logits / 1.5, dim=1)
                                batch_preds = logits.argmax(dim=1).cpu()
                                confidences = probabilities.max(dim=1)[0].cpu()
                                
                                clip_predictions.extend(batch_preds)
                                
                                # Efficient cleanup
                                del image_inputs, image_features, logits
                                
                            except torch.cuda.OutOfMemoryError:
                                print(f"[WARNING] CUDA out of memory, reducing batch size")
                                # Try with smaller batch size
                                for j in range(0, len(batch_rois), max(1, clip_batch_size // 2)):
                                    mini_batch = batch_rois[j:j+max(1, clip_batch_size // 2)]
                                    image_inputs = data_processor(images=mini_batch, return_tensors="pt").to(clip_device)
                                    image_features = clip_model.get_image_features(**image_inputs)
                                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                                    logits = torch.matmul(image_features, text_features.T) * clip_model.logit_scale.exp()
                                    batch_preds = logits.argmax(dim=1).cpu()
                                    clip_predictions.extend(batch_preds)
                                    del image_inputs, image_features, logits
                                    
                        # Conservative memory management
                        if clip_device.type == 'cuda' and i % (clip_batch_size * 4) == 0:
                            torch.cuda.empty_cache()
                            
            except Exception as cuda_error:
                print(f"[ERROR] CUDA processing failed: {cuda_error}")
                print(f"[FALLBACK] Switching to CPU processing")
                
                # Switch to CPU processing
                cpu_device = torch.device('cpu')
                cpu_model = clip_model.cpu()
                
                clip_predictions = []
                for i in range(0, len(clip_rois), clip_batch_size):
                    batch_rois = clip_rois[i:i+clip_batch_size]
                    
                    with torch.no_grad():
                        image_inputs = data_processor(images=batch_rois, return_tensors="pt").to(cpu_device)
                        image_features = cpu_model.get_image_features(**image_inputs)
                        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                        
                        # Move text features to CPU
                        cpu_text_features = text_features.cpu()
                        logits = torch.matmul(image_features, cpu_text_features.T) * cpu_model.logit_scale.exp()
                        batch_preds = logits.argmax(dim=1)
                        clip_predictions.extend(batch_preds)
                        
                        del image_inputs, image_features, logits
                
                # Move model back to GPU if possible
                try:
                    clip_model.to(clip_device)
                    print(f"[INFO] Model moved back to {clip_device}")
                except:
                    print(f"[WARNING] Could not move model back to GPU, staying on CPU")
                    clip_device = torch.device('cpu')
        
        # Update uncertain cases with CLIP results and cache for tracked objects
        for j, roi_idx in enumerate(clip_indices):
            clip_pred = clip_predictions[j].item() if hasattr(clip_predictions[j], 'item') else clip_predictions[j]
            predicted_class = index_to_enum[clip_pred]
            expected_labels[roi_idx] = predicted_class
            
            # Cache result for tracked objects
            _, _, bbox = roi_map[roi_idx]
            tracking_id = bbox.get("id")
            if tracking_id is not None:
                _clip_tracking_cache[tracking_id] = predicted_class
        
        del text_features
        torch.cuda.empty_cache() if clip_device.type == 'cuda' else None
    else:
        print(f"[HEURISTIC] Using heuristic-only classification for all {len(all_rois)} ROIs")
    
    t_heur_end = time.time()
    print(f"[TIMING] Heuristic filtering: {t_heur_end - t_heur_start:.3f}s")
    # --- CLIP inference ---
    t_clip_start = time.time()
    t2 = time.time()
    if not all_rois:
        t3 = time.time(); print(f"[TIMING] Unified CLIP call: {t3-t2:.3f}s")
        return [{"objects": []} for _ in frames]

    # 3. RELAXED scoring and filtering - keep more objects
    t4 = time.time()
    frame_results = [{"objects": []} for _ in frames]  # Pre-initialize with empty objects
    frame_label_count = [{} for _ in frames]
    
    # RELAXED scoring - accept more object types and lower thresholds
    for idx, (frame_idx, obj_idx, bbox) in enumerate(roi_map):
        predicted = predicted_labels[idx]
        expected = expected_labels[idx]
        label_count = frame_label_count[frame_idx].get(expected, 0)
        
        score = BBoxScore(
            predicted_label=predicted,
            expected_label=expected,
            occurrence=label_count
        )
        
        # RELAXED acceptance - keep more objects including lower scoring ones
        if score.points > -0.5:  # Changed from 0.0 to -0.5 to keep more objects
            # Update bbox class_id if needed
            if score.expected_label in EXPECTED_LABEL_TO_CLASS_ID:
                bbox["class_id"] = EXPECTED_LABEL_TO_CLASS_ID[score.expected_label]
            
            frame_results[frame_idx]["objects"].append(bbox)
            frame_label_count[frame_idx][expected] = label_count + 1
    
    t5 = time.time(); print(f"[TIMING] Assign results to frames: {t5-t4:.3f}s")
    t_assign_end = time.time()
    print(f"[TIMING] Total batch_evaluate_frame_filter: {time.time()-t0:.3f}s")
    t_end = time.time()
    print(f"[TIMING] batch_evaluate_frame_filter: total processing time = {t_end - t0:.3f}s")
    return frame_results
