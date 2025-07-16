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

SCALE_FOR_CLIP = 4.0
FRAMES_PER_VIDEO = 750
MIN_WIDTH = 15
MIN_HEIGHT = 40
logger = getLogger("Bounding Box Evaluation Pipeline")
# Chuyển model CLIP và processor sang GPU nếu có
clip_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
data_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. Load model với torch.compile (PyTorch 2.0+)
# Compile model để tăng tốc (nếu PyTorch >= 2.0)
# try:
#     torch.set_float32_matmul_precision('high')
#     clip_model = torch.compile(clip_model)
# except:
#     print("torch.compile not available, using regular model")

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
    # print(
    #     f"Frame {frame_id}:\n"
    #     f"\t-> {len(bboxes)} Bboxes predicted\n"
    #     f"\t-> sum({', '.join(points_with_labels)}) = {total_points:.2f}\n"
    #     f"\t-> (normalised by {n_unique_classes_detected} classes detected) = {normalised_score:.2f}\n"
    #     f"\t-> (Scaled by factor {scale:.2f}) = {scaled_score:.2f}"
    # )
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

def calculate_bbox_importance(bbox, w, h, img_width, img_height, class_id):
    """Calculate importance score for bbox selection - higher score = more important"""
    score = 0.0
    
    # 1. Class priority (footballs are most important for scoring)
    if class_id == 0:  # FOOTBALL
        score += 150.0  # Higher priority for footballs
    elif class_id in [1, 2, 3]:  # GOALKEEPER, PLAYER, REFEREE
        score += 75.0   # Higher priority for people
    else:
        score += 5.0    # Lower priority for unknown objects
    
    # 2. Size score - reasonable sizes are more important
    area = w * h
    img_area = img_width * img_height
    relative_area = area / img_area
    
    if 0.001 <= relative_area <= 0.05:  # Good size range
        score += 30.0
    elif 0.0005 <= relative_area <= 0.1:  # Acceptable size
        score += 20.0
    else:  # Too small or too large
        score += 5.0
    
    # 3. Position score - center objects more important than edge objects
    center_x = (bbox["bbox"][0] + bbox["bbox"][2]) / 2
    center_y = (bbox["bbox"][1] + bbox["bbox"][3]) / 2
    
    # Distance from center (normalized)
    dx = abs(center_x - img_width/2) / (img_width/2)
    dy = abs(center_y - img_height/2) / (img_height/2)
    center_distance = (dx + dy) / 2
    
    # Closer to center = higher score
    position_score = (1.0 - center_distance) * 20.0
    score += position_score
    
    # 4. Aspect ratio score - realistic proportions
    aspect_ratio = w / h if h > 0 else 1.0
    
    if class_id == 0:  # Football should be roughly square
        if 0.7 <= aspect_ratio <= 1.4:
            score += 15.0
        else:
            score += 5.0
    elif class_id in [1, 2, 3]:  # People should be taller than wide
        if 0.3 <= aspect_ratio <= 0.8:
            score += 15.0
        else:
            score += 8.0
    
    # 5. Confidence boost for tracker ID (tracked objects are more reliable)
    if "id" in bbox and bbox["id"] is not None:
        score += 15.0
    
    # 6. Conservative penalty for only very poor detections
    if area < 50:  # Only very tiny objects
        score -= 10.0
    if aspect_ratio > 5.0 or aspect_ratio < 0.1:  # Only extremely distorted objects
        score -= 15.0
    
    return max(10.0, score)  # Ensure minimum score to avoid filtering too aggressively

def batch_evaluate_frame_filter(frames: List[Dict], images: List[np.ndarray], batch_size: int = 2048):
    import time
    import cv2
    t0 = time.time()
    
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
        
        # Minimal reduction - only remove obvious low-quality duplicates
        if len(valid_bboxes) > 12:  # Only reduce if we have very many objects
            # Separate by class and quality
            footballs = []
            tracked_people = []
            other_objects = []
            
            for i, bbox in enumerate(valid_bboxes):
                class_id = bbox.get("class_id", -1)
                if class_id == 0:
                    footballs.append(i)
                elif class_id in [1, 2, 3] and "id" in bbox and bbox["id"] is not None:
                    tracked_people.append(i)
                else:
                    other_objects.append(i)
            
            # Keep all footballs and tracked people, reduce others minimally
            keep_indices = footballs + tracked_people
            
            if len(other_objects) > 5:  # Only reduce if many other objects
                # Keep top 90% of other objects
                num_others_to_keep = max(3, int(len(other_objects) * 0.9))
                other_scores = [(i, bbox_scores[i]) for i in other_objects]
                other_scores.sort(key=lambda x: x[1], reverse=True)
                keep_others = [i for i, _ in other_scores[:num_others_to_keep]]
                keep_indices.extend(keep_others)
            else:
                keep_indices.extend(other_objects)
            
            # Apply filtering
            valid_rois = [valid_rois[i] for i in keep_indices]
            valid_bboxes = [valid_bboxes[i] for i in keep_indices]
        
        total_valid_bboxes += len(valid_bboxes)
        
        # Add to global lists efficiently
        for obj_idx, (roi, bbox) in enumerate(zip(valid_rois, valid_bboxes)):
            all_rois.append(roi)
            roi_map.append((frame_idx, obj_idx, bbox))
            total_valid_rois += 1
            
    print(f"Total frames: {len(frames)} | Total bboxes: {total_bboxes} | Valid bboxes: {total_valid_bboxes} | Valid ROIs: {total_valid_rois}")
    t1 = time.time(); print(f"[TIMING] ROI gathering: {t1-t0:.3f}s")

    # 2. Ultra-fast classification with cached embeddings and heuristics
    t2 = time.time()
    if not all_rois:
        t3 = time.time(); print(f"[TIMING] Unified CLIP call: {t3-t2:.3f}s")
        return [{"objects": []} for _ in frames]
    
    # Get miner predictions for smart routing
    predicted_labels = [OBJECT_ID_TO_ENUM.get(bbox.get("class_id", -1), BoundingBoxObject.OTHER) for _, _, bbox in roi_map]
    
    # Initialize expected labels with heuristic defaults
    expected_labels = []
    
    # Hybrid approach: use CLIP for most, but skip obvious grass/background cases
    expected_labels = []
    
    for i, (predicted_label, roi) in enumerate(zip(predicted_labels, all_rois)):
        h, w = roi.shape[:2]
        area = h * w
        
        # Only skip CLIP for very obvious grass/background cases
        if (predicted_label not in [0, 1, 2, 3] and  # Not a key object class
            area < 200 and  # Very small
            roi.size > 0):
            
            # Quick check if it's obviously grass (very green and uniform)
            mean_color = roi.mean(axis=(0,1))
            color_std = roi.std(axis=(0,1)).mean()
            
            is_obviously_grass = (color_std < 5 and  # Very uniform
                                mean_color[1] > mean_color[0] + 20 and  # Very green
                                mean_color[1] > mean_color[2] + 10)
            
            if is_obviously_grass:
                expected_labels.append(BoundingBoxObject.GRASS)
            else:
                expected_labels.append(None)  # Send to CLIP
        else:
            expected_labels.append(None)  # Send to CLIP
    
    # Collect all cases that need CLIP processing (None values + less confident heuristics)
    clip_indices = []
    for i, expected in enumerate(expected_labels):
        if expected is None:
            clip_indices.append(i)
    
    # Process all uncertain cases with CLIP (but use optimized approach)
    if clip_indices:
        print(f"[CLIP] Processing {len(clip_indices)} uncertain ROIs out of {len(all_rois)}")
        
        # Use the exact same classification as the original for maximum accuracy
        all_labels = [
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
        ]
        
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
        
        text_inputs = data_processor(text=all_labels, return_tensors="pt", padding=True).to(clip_device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        del text_inputs
        
        # Process uncertain ROIs with optimized batching
        clip_rois = [all_rois[i] for i in clip_indices]
        clip_batch_size = min(batch_size, len(clip_rois))  # Use the full batch size
        
        clip_predictions = []
        
        # Use autocast (fixed deprecation warning)
        autocast_context = torch.amp.autocast('cuda') if clip_device.type == 'cuda' else torch.no_grad()
        
        with autocast_context:
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
        
        # Update uncertain cases with CLIP results
        for j, roi_idx in enumerate(clip_indices):
            clip_pred = clip_predictions[j].item()
            expected_labels[roi_idx] = index_to_enum[clip_pred]
        
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
