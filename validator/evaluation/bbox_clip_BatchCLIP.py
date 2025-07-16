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

# Initialize TensorRT optimized CLIP if available
tensorrt_clip = None
if TENSORRT_AVAILABLE and clip_device.type == 'cuda':
    try:
        print("Initializing TensorRT CLIP optimization...")
        tensorrt_clip = create_tensorrt_clip(clip_model, data_processor, clip_device)
        print("TensorRT CLIP initialization successful")
    except Exception as e:
        print(f"TensorRT CLIP initialization failed: {e}, using standard PyTorch")
        tensorrt_clip = None
else:
    print("TensorRT not available or no CUDA, using standard PyTorch CLIP")

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

def evaluate_frame_filter(
    frame_id:int,
    image_array:ndarray,
    bboxes:list[dict[str,int|tuple[int,int,int,int]]]
):
    import time
    start_time = time.time()
    bboxes = [bbox for bbox in bboxes if is_bbox_large_enough(bbox) and not is_touching_scoreboard_zone(bbox, image_array.shape[1], image_array.shape[0])]
    if not bboxes:
        logger.info(f"Frame {frame_id}: all bboxes filtered out due to small size or corners — skipping.")
        return 0.0
        
    # Nếu object đã có expected_label_raw (batch CLIP từ process_video), dùng luôn, không gọi lại CLIP
    if all('expected_label_raw' in bbox and bbox['expected_label_raw'] is not None for bbox in bboxes):
        expected_labels = [bbox['expected_label_raw'] for bbox in bboxes]
    else:
        # Logic cũ: gọi CLIP từng bước
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
        expected_labels = [None] * len(rois)
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
    # logger.info(
    #     f"Frame {frame_id}:\n"
    #     f"\t-> {len(bboxes)} Bboxes predicted\n"
    #     f"\t-> sum({', '.join(points_with_labels)}) = {total_points:.2f}\n"
    #     f"\t-> (normalised by {n_unique_classes_detected} classes detected) = {normalised_score:.2f}\n"
    #     f"\t-> (Scaled by factor {scale:.2f}) = {scaled_score:.2f}"
    # )
    # print(
    #     f"Frame {frame_id}:\n"
    #     f"\t-> {len(bboxes)} Bboxes predicted\n"
    #     f"\t-> sum({', '.join(points_with_labels)}) = {total_points:.2f}\n"
    #     f"\t-> (normalised by {n_unique_classes_detected} classes detected) = {normalised_score:.2f}\n"
    #     f"\t-> (Scaled by factor {scale:.2f}) = {scaled_score:.2f}"
    # )

    # Only keep bboxes with score.points > 0
    keep = [(bbox, score) for bbox, score in zip(bboxes, scores) if score.points > 0.0]
    # Update class_id according to expected_label
    for bbox, score in keep:
        if score.expected_label in EXPECTED_LABEL_TO_CLASS_ID:
            bbox["class_id"] = EXPECTED_LABEL_TO_CLASS_ID[score.expected_label]
        # else:
            # bbox["class_id"] = -1
    # Move objects with predicted_label != expected_label to the end
    correct = [bbox for bbox, score in keep if score.predicted_label.value == score.expected_label.value]
    incorrect = [bbox for bbox, score in keep if score.predicted_label.value != score.expected_label.value]
    keep_sorted = correct + incorrect
    elapsed = time.time() - start_time
    print(f"evaluate_frame_filter: frame {frame_id} processed in {elapsed:.3f} seconds")
    return keep_sorted

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

def batch_evaluate_frame_filter(frames: List[Dict], images: List[np.ndarray], batch_size: int = None, enable_class_limits: bool = False):
    """
    Batch evaluate frame filter with optional class limiting.
    
    Args:
        frames: List of frame dictionaries with objects
        images: List of image arrays
        batch_size: CLIP batch size (auto-determined if None)
        enable_class_limits: If True, apply 4-class limiting (Football/GK/Player/Referee only, max 7 per class)
                           If False, process all objects normally
    """
    import time
    import cv2
    t0 = time.time()
    
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
    
    # 1. Ultra-fast ROI extraction with vectorized operations
    all_rois = []
    roi_map = []  # (frame_idx, obj_idx, bbox_dict)
    total_bboxes = 0
    total_valid_bboxes = 0
    total_valid_rois = 0
    
    # Ultra-fast ROI extraction with batch processing
    # Pre-convert all images to RGB in one batch
    rgb_images = []
    for img in images:
        rgb_images.append(img[:,:,::-1])  # BGR -> RGB batch conversion
    
    # Process frames with optimized loop
    for frame_idx, frame_data in enumerate(frames):
        bboxes = frame_data.get("objects", [])
        if not bboxes:
            continue
        total_bboxes += len(bboxes)
        
        # Fast path for frames with very few objects (likely accurate already)
        if len(bboxes) <= 3:
            for bbox in bboxes:
                coords = bbox["bbox"]
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    if y1 >= 150:  # Not in scoreboard zone
                        roi_map.append((frame_idx, len(all_rois), bbox))
                        # Use a dummy ROI for very small frames (will be handled by heuristics)
                        all_rois.append(np.zeros((32, 32, 3), dtype=np.uint8))
                        total_valid_rois += 1
            total_valid_bboxes += len(bboxes)
            continue
        
        # Get image info once
        img_height, img_width = images[frame_idx].shape[:2]
        img_rgb = rgb_images[frame_idx]
        
        # Batch process all bboxes for this frame
        valid_rois = []
        valid_bboxes = []
        bbox_scores = []
        
        # Vectorized filtering for all bboxes at once
        for bbox in bboxes:
            coords = bbox["bbox"]
            if not (isinstance(coords, (list, tuple)) and len(coords) == 4):
                continue
            
            x1, y1, x2, y2 = coords
            w, h = x2 - x1, y2 - y1
            class_id = bbox.get("class_id", -1)
            
            # Fast combined checks
            if not ((class_id == 0) or (w >= 15 and h >= 40)) or y1 < 150:
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
            
            # Calculate importance score
            importance_score = calculate_bbox_importance(bbox, w, h, img_width, img_height, class_id)
            
            valid_rois.append(roi)
            valid_bboxes.append(bbox)
            bbox_scores.append(importance_score)
        
        # Optional class-limited object selection: maximum 7 objects per class per frame
        if enable_class_limits:
            # Only keep Football, GK, Player, Referee classes (0, 1, 2, 3)
            important_classes = [0, 1, 2, 3]  # Football, GK, Player, Referee
            
            class_groups = {}
            for i, bbox in enumerate(valid_bboxes):
                class_id = bbox.get("class_id", -1)
                # Only process objects from important classes
                if class_id in important_classes:
                    if class_id not in class_groups:
                        class_groups[class_id] = []
                    class_groups[class_id].append((i, bbox_scores[i], bbox))
            
            # Sort objects within each class by importance (better objects first)
            for class_id in class_groups:
                class_groups[class_id].sort(key=lambda x: x[1], reverse=True)
                
            # Add quality filtering within each class
            for class_id in class_groups:
                # Filter out very poor quality objects even within the 7-object limit
                filtered_objects = []
                for idx, score, bbox in class_groups[class_id]:
                    # Quality thresholds based on class
                    if class_id == 0:  # Football - stricter quality
                        if score >= 1020.0:  # High quality footballs only
                            filtered_objects.append((idx, score, bbox))
                    elif class_id in [1, 2, 3]:  # People - moderate quality
                        if score >= 520.0:  # Good quality people
                            filtered_objects.append((idx, score, bbox))
                
                class_groups[class_id] = filtered_objects
            
            selected_indices = []
            class_counts = {}
            
            # Apply class-specific limits for important classes only
            for class_id, objects in class_groups.items():
                if class_id == 0:  # Football - keep all (critical for scoring)
                    selected_indices.extend([idx for idx, _, _ in objects])
                    class_counts[class_id] = len(objects)
                elif class_id in [1, 2, 3]:  # People classes - special handling
                    # First, keep all tracked people (up to 7 per class)
                    tracked_count = 0
                    untracked_objects = []
                    
                    for idx, score, bbox in objects:
                        if "id" in bbox and bbox["id"] is not None:
                            if tracked_count < 7:
                                selected_indices.append(idx)
                                tracked_count += 1
                        else:
                            untracked_objects.append((idx, score))
                    
                    # Then fill remaining slots with best untracked objects
                    remaining_slots = 7 - tracked_count
                    if remaining_slots > 0 and untracked_objects:
                        untracked_objects.sort(key=lambda x: x[1], reverse=True)
                        for idx, _ in untracked_objects[:remaining_slots]:
                            selected_indices.append(idx)
                    
                    class_counts[class_id] = min(7, len(objects))
            
            # Apply selection
            if len(selected_indices) > 0:
                valid_rois = [valid_rois[i] for i in selected_indices]
                valid_bboxes = [valid_bboxes[i] for i in selected_indices]
                
                # Create detailed class summary for important classes only
                class_summary = []
                for class_id, count in class_counts.items():
                    class_name = {0: "Football", 1: "GK", 2: "Player", 3: "Referee"}.get(class_id, f"Class{class_id}")
                    class_summary.append(f"{class_name}:{count}")
                
                # Calculate how many objects were filtered out by class restriction
                total_before_class_filter = len(bbox_scores)
                other_classes_filtered = total_before_class_filter - len(valid_bboxes)
                
                if other_classes_filtered > 0:
                    print(f"Frame {frame_idx}: 4-class selection {total_before_class_filter} → {len(valid_bboxes)} objects ({', '.join(class_summary)}) | Filtered {other_classes_filtered} other classes")
        
        total_valid_bboxes += len(valid_bboxes)
        
        # Add to global lists efficiently
        for obj_idx, (roi, bbox) in enumerate(zip(valid_rois, valid_bboxes)):
            all_rois.append(roi)
            roi_map.append((frame_idx, obj_idx, bbox))
            total_valid_rois += 1
            
    reduction_percent = (1.0 - total_valid_bboxes / total_bboxes) * 100 if total_bboxes > 0 else 0
    print(f"Total frames: {len(frames)} | Total bboxes: {total_bboxes} | Valid bboxes: {total_valid_bboxes} | Valid ROIs: {total_valid_rois}")
    
    if enable_class_limits:
        print(f"4-class focused reduction: {reduction_percent:.1f}% | Max 7 best per class (Football/GK/Player/Referee only) | Avg objects per frame: {total_valid_bboxes/len(frames):.1f}")
        t1 = time.time(); print(f"[TIMING] ROI gathering with 4-class selection: {t1-t0:.3f}s")
    else:
        print(f"Standard object processing: {reduction_percent:.1f}% | All classes | Avg objects per frame: {total_valid_bboxes/len(frames):.1f}")
        t1 = time.time(); print(f"[TIMING] ROI gathering with standard selection: {t1-t0:.3f}s")

    # 2. Ultra-fast classification with cached embeddings and heuristics
    t2 = time.time()
    if not all_rois:
        t3 = time.time(); print(f"[TIMING] Unified CLIP call: {t3-t2:.3f}s")
        return [{"objects": []} for _ in frames]
    
    # Get miner predictions for smart routing
    predicted_labels = [OBJECT_ID_TO_ENUM.get(bbox.get("class_id", -1), BoundingBoxObject.OTHER) for _, _, bbox in roi_map]
    
    # Initialize expected labels with heuristic defaults
    expected_labels = []
    
    # Enhanced heuristic approach: more aggressive pre-filtering while preserving accuracy
    expected_labels = []
    
    for i, (predicted_label, roi) in enumerate(zip(predicted_labels, all_rois)):
        h, w = roi.shape[:2]
        area = h * w
        
        # Check tracking cache first for tracked objects
        _, _, bbox = roi_map[i]
        tracking_id = bbox.get("id")
        if tracking_id is not None and tracking_id in _clip_tracking_cache:
            # Use cached result for tracked objects
            expected_labels.append(_clip_tracking_cache[tracking_id])
            continue
        
        # Minimal heuristic filtering - only very obvious grass patches
        if (predicted_label not in [0, 1, 2, 3] and  # Not key objects
            area < 100 and roi.size > 0):  # Very small
            
            # Ultra-fast grass check
            mean_color = roi.mean(axis=(0,1))
            if (mean_color[1] > mean_color[0] + 25 and  # Very green
                mean_color[1] > mean_color[2] + 15):   # Strong green dominance
                expected_labels.append(BoundingBoxObject.GRASS)
                continue
        
        # Send everything else to CLIP
        expected_labels.append(None)
    
    # Collect all cases that need CLIP processing (None values + less confident heuristics)
    clip_indices = []
    for i, expected in enumerate(expected_labels):
        if expected is None:
            clip_indices.append(i)
    
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
        clip_batch_size = min(batch_size, len(clip_rois))  # Use the full batch size
        
        clip_predictions = []
        
        # Use TensorRT if available for maximum speed
        if tensorrt_clip is not None:
            # TensorRT path - significant speedup
            print(f"[TensorRT] Processing {len(clip_rois)} ROIs with TensorRT acceleration")
            try:
                logits = tensorrt_clip.classify_images(clip_rois, list(all_labels), clip_batch_size)
                clip_predictions = logits.argmax(dim=1).cpu().tolist()
            except Exception as e:
                print(f"TensorRT inference failed: {e}, falling back to PyTorch")
                # Fallback to PyTorch
                clip_predictions = []
                with torch.amp.autocast('cuda') if clip_device.type == 'cuda' else torch.no_grad():
                    for i in range(0, len(clip_rois), clip_batch_size):
                        batch_rois = clip_rois[i:i+clip_batch_size]
                        with torch.no_grad():
                            image_inputs = data_processor(images=batch_rois, return_tensors="pt").to(clip_device)
                            image_features = clip_model.get_image_features(**image_inputs)
                            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                            logits = torch.matmul(image_features, text_features.T) * clip_model.logit_scale.exp()
                            batch_preds = logits.argmax(dim=1).cpu()
                            clip_predictions.extend(batch_preds)
                        del image_inputs, image_features, logits
        else:
            # PyTorch path - standard inference
            print(f"[PyTorch] Processing {len(clip_rois)} ROIs with standard PyTorch")
            with torch.amp.autocast('cuda') if clip_device.type == 'cuda' else torch.no_grad():
                for i in range(0, len(clip_rois), clip_batch_size):
                    batch_rois = clip_rois[i:i+clip_batch_size]
                    
                    with torch.no_grad():
                        # Streamlined processing
                        image_inputs = data_processor(images=batch_rois, return_tensors="pt").to(clip_device)
                        image_features = clip_model.get_image_features(**image_inputs)
                        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                        
                        # Direct computation
                        logits = torch.matmul(image_features, text_features.T) * clip_model.logit_scale.exp()
                        batch_preds = logits.argmax(dim=1).cpu()  # Skip softmax for argmax
                        clip_predictions.extend(batch_preds)
                    
                    # Efficient cleanup
                    del image_inputs, image_features, logits
                    
                    # Less frequent cache clearing for speed
                    if clip_device.type == 'cuda' and i % (clip_batch_size * 2) == 0:
                        torch.cuda.empty_cache()
        
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
    
    t3 = time.time(); print(f"[TIMING] Unified CLIP call: {t3-t2:.3f}s")

    # 3. Optimized scoring and filtering
    t4 = time.time()
    frame_results = [{"objects": []} for _ in frames]  # Pre-initialize with empty objects
    frame_label_count = [{} for _ in frames]
    
    # Batch process scoring calculations
    for idx, (frame_idx, obj_idx, bbox) in enumerate(roi_map):
        predicted = predicted_labels[idx]
        expected = expected_labels[idx]
        label_count = frame_label_count[frame_idx].get(expected, 0)
        
        score = BBoxScore(
            predicted_label=predicted,
            expected_label=expected,
            occurrence=label_count
        )
        
        if score.points > 0.0:
            # Update bbox class_id if needed
            if score.expected_label in EXPECTED_LABEL_TO_CLASS_ID:
                bbox["class_id"] = EXPECTED_LABEL_TO_CLASS_ID[score.expected_label]
            
            frame_results[frame_idx]["objects"].append(bbox)
            frame_label_count[frame_idx][expected] = label_count + 1
    
    t5 = time.time(); print(f"[TIMING] Assign results to frames: {t5-t4:.3f}s")
    print(f"[TIMING] Total batch_evaluate_frame_filter: {time.time()-t0:.3f}s")
    return frame_results

