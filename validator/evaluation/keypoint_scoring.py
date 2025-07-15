#!/usr/bin/env python3
import sys
from pathlib import Path
# Add miner directory to Python path
miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

import cv2
import numpy as np
import json
import argparse
from miner.sports.configs.soccer import SoccerPitchConfiguration
from collections import defaultdict

def preprocess_keypoints(keypoints):
    """
    Preprocess keypoints from the ground truth format to the expected format.
    
    Parameters:
    keypoints (list): Flat list of keypoints [x1, y1, x2, y2, ...]
    
    Returns:
    np.ndarray: Array of keypoint coordinates in the format [[x1, y1], [x2, y2], ...]
    """
    return np.array(keypoints).reshape(-1, 2)

def get_valid_keypoints(keypoints):
    """
    Get valid keypoints (non-zero coordinates) while maintaining their original indices.
    
    Parameters:
    keypoints (np.ndarray): Array of keypoint coordinates.
    
    Returns:
    np.ndarray: Valid keypoints.
    np.ndarray: Indices of valid keypoints.
    """
    valid_mask = ~(np.all(keypoints == 0, axis=1))
    valid_indices = np.where(valid_mask)[0]
    return keypoints[valid_mask], valid_indices

def normalize_points(points):
    """
    Normalize points by subtracting the mean and scaling to unit standard deviation.
    
    Parameters:
    points (np.ndarray): Array of point coordinates.
    
    Returns:
    np.ndarray: Normalized points.
    np.ndarray: Normalization matrix.
    """
    mean = np.mean(points, axis=0)
    std = np.std(points)
    norm_points = (points - mean) / std
    norm_matrix = np.array([[1/std, 0, -mean[0]/std],
                            [0, 1/std, -mean[1]/std],
                            [0, 0, 1]])
    return norm_points, norm_matrix

def estimate_homography_ransac(keypoints, video_width, video_height, ransac_thresh_ratio=0.01):
    """
    Estimate a homography from keypoints in image coordinates to pitch vertices scaled to pixel coordinates using RANSAC.
    
    This function uses the vertices defined in the SoccerPitchConfiguration and scales them to match the image dimensions.
    
    Parameters:
      keypoints (np.ndarray): An (N, 2) array of detected keypoints in pixel coordinates.
      video_width (int): Width of the video frame in pixels.
      video_height (int): Height of the video frame in pixels.
      ransac_thresh_ratio (float): The reprojection threshold for RANSAC as a ratio of video width.
    
    Returns:
      H (np.ndarray or None): The 3x3 homography matrix if computed, else None.
      inlier_ratio (float): The ratio of inliers to total valid keypoints.
      avg_reprojection_error (float): The average reprojection error for all points in pixels.
    """
    # Load the soccer pitch configuration and extract vertices.
    pitch_config = SoccerPitchConfiguration()
    pitch_vertices = np.array(pitch_config.vertices, dtype=np.float32)

    # Scale pitch vertices to match image dimensions
    pitch_width = pitch_config.width
    pitch_height = pitch_config.length
    scale_x = video_width / pitch_width
    scale_y = video_height / pitch_height
    scaled_pitch_vertices = pitch_vertices * np.array([scale_x, scale_y])

    # Get valid keypoints and their indices
    valid_keypoints, valid_indices = get_valid_keypoints(keypoints)

    # Ensure at least 4 valid points are available.
    if valid_keypoints.shape[0] < 4:
        return None, 0.0, float('inf')

    # Match the scaled pitch vertices to the valid keypoints
    valid_pitch_vertices = scaled_pitch_vertices[valid_indices]

    # Calculate RANSAC threshold based on video width
    ransac_thresh = video_width * ransac_thresh_ratio
    ransac_thresh = max(ransac_thresh, 12)  # Ensure a minimum threshold of 12 pixels

    # Use OpenCV's RANSAC implementation to estimate the homography
    H, mask = cv2.findHomography(valid_keypoints, valid_pitch_vertices, cv2.RANSAC, ransac_thresh)
    
    if H is None or mask is None:
        return None, 0.0, float('inf')
    
    inlier_ratio = float(np.sum(mask)) / mask.size

    # Calculate reprojection errors in pixel space
    reprojected_points = cv2.perspectiveTransform(valid_keypoints.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(reprojected_points - valid_pitch_vertices, axis=1)
    avg_reprojection_error = np.mean(errors)

    return H, inlier_ratio, avg_reprojection_error

def detect_scene_transitions(frames, large_jump_threshold=20, transition_threshold=0.5):
    """
    Detect scene transitions (camera cuts, replays, close-ups) in the video.
    
    Parameters:
    frames (dict): Dictionary of frame data.
    large_jump_threshold (float): Threshold for considering a player jump as large.
    transition_threshold (float): Threshold ratio of players with large jumps to consider a scene transition.
    
    Returns:
    list: List of frame IDs where scene transitions occur.
    dict: Dictionary mapping frame IDs to segment IDs.
    """
    frame_ids = sorted([int(fid) for fid in frames.keys()])
    transitions = []
    frame_to_segment = {}
    current_segment = 0
    
    # Track player positions across frames
    player_positions = {}
    
    for i, frame_id in enumerate(frame_ids):
        frame_id_str = str(frame_id)
        frame_to_segment[frame_id_str] = current_segment
        
        # Skip first frame as we need a previous frame for comparison
        if i == 0:
            # Initialize player positions for first frame
            for obj in frames[frame_id_str].get("objects", []):
                if obj.get("class_id") == 2:  # Player
                    pid = obj["id"]
                    bbox = obj["bbox"]
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    player_positions[pid] = center
            continue
        
        # Count large jumps in current frame
        large_jumps = 0
        total_players = 0
        current_positions = {}
        
        for obj in frames[frame_id_str].get("objects", []):
            if obj.get("class_id") == 2:  # Player
                pid = obj["id"]
                bbox = obj["bbox"]
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                current_positions[pid] = center
                
                if pid in player_positions:
                    prev_center = player_positions[pid]
                    jump = np.linalg.norm(np.array(center) - np.array(prev_center))
                    
                    if jump > large_jump_threshold:
                        large_jumps += 1
                
                total_players += 1
        
        # Update player positions for next frame
        player_positions = current_positions
        
        # Check if this is a scene transition
        if total_players > 0 and large_jumps / total_players >= transition_threshold:
            transitions.append(frame_id)
            current_segment += 1
    
    return transitions, frame_to_segment

def check_keypoint_stability(frames, frame_to_segment, video_width, video_height):
    """
    Check if keypoints maintain stable relative positions on the pitch within segments.
    
    Parameters:
    frames (dict): Dictionary of frame data.
    frame_to_segment (dict): Dictionary mapping frame IDs to segment IDs.
    video_width (int): Width of the video frame in pixels.
    video_height (int): Height of the video frame in pixels.
    
    Returns:
    dict: Dictionary with segment stability scores.
    """
    # Group frames by segment
    segments = defaultdict(list)
    for frame_id, segment_id in frame_to_segment.items():
        segments[segment_id].append(frame_id)
    
    stability_scores = {}
    
    for segment_id, segment_frames in segments.items():
        if len(segment_frames) < 2:
            stability_scores[segment_id] = {
                'keypoint_stability': 1.0,  # Default score for single-frame segments
                'homography_stability': 1.0,
                'frames_analyzed': len(segment_frames)
            }
            continue
        
        # Calculate homography for each frame in the segment
        homographies = {}
        valid_keypoints_map = {}
        
        for frame_id in segment_frames:
            if 'keypoints' not in frames[frame_id]:
                continue
                
            keypoints = preprocess_keypoints(frames[frame_id]['keypoints'])
            H, inlier_ratio, avg_reprojection_error = estimate_homography_ransac(
                keypoints, video_width, video_height
            )
            
            if H is not None:
                homographies[frame_id] = H
                valid_keypoints, valid_indices = get_valid_keypoints(keypoints)
                valid_keypoints_map[frame_id] = (valid_keypoints, valid_indices)
        
        # Skip segments with fewer than 2 valid homographies
        if len(homographies) < 2:
            stability_scores[segment_id] = {
                'keypoint_stability': 0.5,  # Neutral score for segments with insufficient data
                'homography_stability': 0.5,
                'frames_analyzed': len(homographies)
            }
            continue
        
        # Calculate homography stability
        homography_diffs = []
        frame_ids = sorted(homographies.keys())
        
        for i in range(1, len(frame_ids)):
            prev_H = homographies[frame_ids[i-1]]
            curr_H = homographies[frame_ids[i]]
            
            # Calculate Frobenius norm of difference between normalized homographies
            norm_prev_H = prev_H / np.linalg.norm(prev_H)
            norm_curr_H = curr_H / np.linalg.norm(curr_H)
            diff = np.linalg.norm(norm_prev_H - norm_curr_H)
            homography_diffs.append(diff)
        
        avg_homography_diff = np.mean(homography_diffs)
        homography_stability = np.exp(-avg_homography_diff)  # Convert to 0-1 score (higher is better)
        
        # Calculate keypoint stability (relative positions)
        keypoint_diffs = []
        
        for i in range(1, len(frame_ids)):
            prev_frame_id = frame_ids[i-1]
            curr_frame_id = frame_ids[i]
            
            if prev_frame_id not in valid_keypoints_map or curr_frame_id not in valid_keypoints_map:
                continue
                
            prev_keypoints, prev_indices = valid_keypoints_map[prev_frame_id]
            curr_keypoints, curr_indices = valid_keypoints_map[curr_frame_id]
            
            # Find common keypoints
            common_indices = np.intersect1d(prev_indices, curr_indices)
            
            if len(common_indices) < 2:
                continue
                
            # Get positions of common keypoints
            prev_common = np.array([prev_keypoints[np.where(prev_indices == idx)[0][0]] for idx in common_indices])
            curr_common = np.array([curr_keypoints[np.where(curr_indices == idx)[0][0]] for idx in common_indices])
            
            # Calculate pairwise distances within each frame
            prev_distances = []
            curr_distances = []
            
            for i in range(len(common_indices)):
                for j in range(i+1, len(common_indices)):
                    prev_dist = np.linalg.norm(prev_common[i] - prev_common[j])
                    curr_dist = np.linalg.norm(curr_common[i] - curr_common[j])
                    prev_distances.append(prev_dist)
                    curr_distances.append(curr_dist)
            
            # Calculate relative difference in distances
            if len(prev_distances) > 0:
                prev_distances = np.array(prev_distances)
                curr_distances = np.array(curr_distances)
                rel_diffs = np.abs(prev_distances - curr_distances) / (prev_distances + 1e-6)
                avg_rel_diff = np.mean(rel_diffs)
                keypoint_diffs.append(avg_rel_diff)
        
        if len(keypoint_diffs) > 0:
            avg_keypoint_diff = np.mean(keypoint_diffs)
            keypoint_stability = np.exp(-10 * avg_keypoint_diff)  # Convert to 0-1 score (higher is better)
        else:
            keypoint_stability = 0.5  # Neutral score if no common keypoints
        
        stability_scores[segment_id] = {
            'keypoint_stability': keypoint_stability,
            'homography_stability': homography_stability,
            'frames_analyzed': len(homographies)
        }
    
    return stability_scores

def check_player_plausibility(frames, frame_to_segment, max_speed_pixels=30):
    """
    Check if player movements are physically plausible within continuous segments.
    
    Parameters:
    frames (dict): Dictionary of frame data.
    frame_to_segment (dict): Dictionary mapping frame IDs to segment IDs.
    max_speed_pixels (float): Maximum plausible player movement in pixels per frame.
    
    Returns:
    dict: Dictionary with segment plausibility scores.
    """
    # Group frames by segment
    segments = defaultdict(list)
    for frame_id, segment_id in frame_to_segment.items():
        segments[segment_id].append(frame_id)
    
    plausibility_scores = {}
    
    for segment_id, segment_frames in segments.items():
        # Convert to integers and sort
        segment_frames = sorted([int(fid) for fid in segment_frames])
        segment_frames = [str(fid) for fid in segment_frames]  # Convert back to strings
        
        if len(segment_frames) < 2:
            plausibility_scores[segment_id] = {
                'plausibility_score': 1.0,  # Default score for single-frame segments
                'implausible_movements': 0,
                'total_movements': 0
            }
            continue
        
        # Track player positions across frames
        player_positions = {}
        implausible_movements = 0
        total_movements = 0
        
        # Process frames in order
        for i in range(len(segment_frames)):
            frame_id = segment_frames[i]
            current_positions = {}
            
            # Get player positions in current frame
            for obj in frames[frame_id].get("objects", []):
                if obj.get("class_id") == 2:  # Player
                    pid = obj["id"]
                    bbox = obj["bbox"]
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    current_positions[pid] = center
                    
                    # Check movement if player was in previous frame
                    if i > 0 and pid in player_positions:
                        prev_center = player_positions[pid]
                        movement = np.linalg.norm(np.array(center) - np.array(prev_center))
                        
                        # Only count consecutive frames
                        prev_frame_id = segment_frames[i-1]
                        if int(frame_id) - int(prev_frame_id) <= 2:  # Allow for 1 frame gap
                            total_movements += 1
                            
                            # Lower the threshold to catch more implausible movements
                            if movement > max_speed_pixels:
                                implausible_movements += 1
                                print(f"Implausible movement detected: Player {pid} moved {movement:.2f} pixels between frames {prev_frame_id} and {frame_id}")
            
            # Update player positions for next frame
            player_positions = current_positions
        
        # Calculate plausibility score
        if total_movements > 0:
            plausibility_score = max(0, 1.0 - (implausible_movements / total_movements))
            print(f"Segment {segment_id}: {implausible_movements} implausible movements out of {total_movements} total movements")
        else:
            plausibility_score = 0.5  # Neutral score if no movements to analyze
            print(f"Segment {segment_id}: No player movements to analyze")
        
        plausibility_scores[segment_id] = {
            'plausibility_score': plausibility_score,
            'implausible_movements': implausible_movements,
            'total_movements': total_movements
        }
    
    return plausibility_scores

def score_player_positions(frames, small_jump_threshold=2, medium_jump_threshold=10):
    """
    Compute a consistency score based on the relative movement of players.
    Assumes each frame has an "objects" key, and each object has an "id", "bbox", and "class_id".
    Only considers objects with class_id == 2 (players).

    Returns:
        final_score (float): A score from 0 to 100 (higher is better).
        avg_jump (float): Average jump (in pixels) across all tracked players.
        total_jumps (int): Total number of jumps computed.
        biggest_jump (float): The largest jump observed.
        large_jumps (int): Number of large jumps (potential substitutions or view changes).
    """
    player_tracks = {}
    for frame_id, frame_data in frames.items():
        for obj in frame_data.get("objects", []):
            if obj.get("class_id") == 2:
                bbox = obj["bbox"]
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                pid = obj["id"]
                if pid not in player_tracks:
                    player_tracks[pid] = []
                player_tracks[pid].append((int(frame_id), center))
    
    for pid in player_tracks:
        player_tracks[pid].sort(key=lambda x: x[0])
    
    total_jump = 0
    jump_count = 0
    biggest_jump = 0
    large_jumps = 0
    small_jumps = 0
    medium_jumps = 0
    
    total_score = 0
    max_possible_score = 0

    for pid, track in player_tracks.items():
        for i in range(1, len(track)):
            _, center_prev = track[i-1]
            _, center_curr = track[i]
            jump = np.linalg.norm(np.array(center_curr) - np.array(center_prev))
            
            total_jump += jump
            jump_count += 1
            biggest_jump = max(biggest_jump, jump)
            
            if jump <= small_jump_threshold:
                small_jumps += 1
                total_score += 1  # Full score for small jumps
            elif jump <= medium_jump_threshold:
                medium_jumps += 1
                total_score += 0.5  # Half score for medium jumps (slight penalty)
            else:
                large_jumps += 1
                total_score += 1  # Full score for large jumps (flagged but not penalized)
            
            max_possible_score += 1  # Maximum score possible for each jump

    avg_jump = total_jump / jump_count if jump_count > 0 else 0
    final_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    print(f"Small jumps: {small_jumps}, Medium jumps: {medium_jumps}, Large jumps: {large_jumps}")
    
    return final_score, avg_jump, jump_count, biggest_jump, large_jumps

def calculate_keypoint_score(inlier_ratio, avg_reprojection_error):
    inlier_score = min(inlier_ratio * 100, 100)  # Scale to 0-100
    reprojection_score = max(0, 100 - avg_reprojection_error)  # Inverse scale, lower is better
    return (inlier_score + reprojection_score) / 2

def process_input_file(input_file, video_width, video_height):
    if isinstance(input_file, str): 
        with open(input_file, 'r') as f:
            data = json.load(f)
            
    elif isinstance(input_file, dict):  # Already JSON data
        if "frames" not in input_file:
            data = {"frames": input_file}
        else:
            data = input_file
            
    elif isinstance(input_file, list):
        frames_dict = {}
        for i, frame_data in enumerate(input_file):
            if frame_data:
                # Use frame_number if available, otherwise use index
                frame_id = str(frame_data.get('frame_number', i))
                frames_dict[frame_id] = frame_data
        frames_data={"frames": frames_dict}
        
    # Convert list format to dictionary format if needed
    frames_data = data['frames']

    # Detect scene transitions and segment the video
    transitions, frame_to_segment = detect_scene_transitions(data['frames'])
    print(f"Detected {len(transitions)} scene transitions")
    print(f"Video segmented into {len(set(frame_to_segment.values()))} continuous segments")
    
    # Check keypoint stability within segments
    stability_scores = check_keypoint_stability(data['frames'], frame_to_segment, video_width, video_height)
    
    # Check player movement plausibility within segments
    plausibility_scores = check_player_plausibility(data['frames'], frame_to_segment)
    
    # Process keypoints for each frame
    results = {}
    valid_frames = 0
    total_inlier_ratio = 0
    total_reprojection_error = 0
    total_keypoint_score = 0
    
    for frame_id, frame_data in data['frames'].items():
        keypoints = preprocess_keypoints(frame_data['keypoints'])
        H, inlier_ratio, avg_reprojection_error = estimate_homography_ransac(keypoints, video_width, video_height)
        
        if H is not None:
            # print(f"Inlier ratio: {inlier_ratio}, Avg reprojection error: {avg_reprojection_error}")
            keypoint_score = calculate_keypoint_score(inlier_ratio, avg_reprojection_error)
            # print(f"Keypoint score: {keypoint_score}")
            # Get segment-specific scores
            segment_id = frame_to_segment.get(frame_id, 0)
            segment_stability = stability_scores.get(segment_id, {'keypoint_stability': 1.0, 'homography_stability': 1.0})
            segment_plausibility = plausibility_scores.get(segment_id, {'plausibility_score': 1.0})
            
            results[frame_id] = {
                'inlier_ratio': float(inlier_ratio),
                'avg_reprojection_error': float(avg_reprojection_error),
                'keypoint_score': keypoint_score,
                'segment_id': segment_id,
                'keypoint_stability': segment_stability['keypoint_stability'],
                'homography_stability': segment_stability['homography_stability'],
                'player_plausibility': segment_plausibility['plausibility_score']
            }
            valid_frames += 1
            total_inlier_ratio += inlier_ratio
            total_reprojection_error += avg_reprojection_error
            total_keypoint_score += keypoint_score
    
    avg_inlier_ratio = total_inlier_ratio / valid_frames if valid_frames > 0 else 0
    avg_reprojection_error = total_reprojection_error / valid_frames if valid_frames > 0 else float('inf')
    avg_keypoint_score = total_keypoint_score / valid_frames if valid_frames > 0 else 0
    
    # Calculate player position consistency score
    player_score, avg_jump, total_jumps, biggest_jump, large_jumps = score_player_positions(data['frames'])
    
    # Calculate average stability and plausibility scores
    avg_keypoint_stability = np.mean([r['keypoint_stability'] for r in results.values()]) if results else 0
    avg_homography_stability = np.mean([r['homography_stability'] for r in results.values()]) if results else 0
    avg_player_plausibility = np.mean([r['player_plausibility'] for r in results.values()]) if results else 0
    
    return results, valid_frames, avg_inlier_ratio, avg_reprojection_error, avg_keypoint_score, player_score, avg_jump, total_jumps, biggest_jump, large_jumps, transitions, avg_keypoint_stability, avg_homography_stability, avg_player_plausibility

def summarize_scores(results):
    inlier_ranges = {
        '0-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, '41-50': 0,
        '51-60': 0, '61-70': 0, '71-80': 0, '81-90': 0, '91-100': 0
    }
    error_ranges = {
        '0-5': 0, '5-10': 0, '10-15': 0, '15-20': 0, '20-25': 0,
        '25-30': 0, '30-35': 0, '35-40': 0, '40-45': 0, '45+': 0
    }
    
    for frame_data in results.values():
        inlier_score = frame_data['inlier_ratio'] * 100
        error_score = frame_data['avg_reprojection_error']
        
        for range_key in inlier_ranges:
            low, high = map(int, range_key.split('-'))
            if low <= inlier_score <= high:
                inlier_ranges[range_key] += 1
                break
        
        for range_key in error_ranges:
            if range_key == '45+':
                if error_score >= 45:
                    error_ranges[range_key] += 1
            else:
                low, high = map(int, range_key.split('-'))
                if low <= error_score < high:
                    error_ranges[range_key] += 1
                    break
    
    return {'inlier_ratio': inlier_ranges, 'avg_reprojection_error': error_ranges}

def calculate_final_score_keypoints(keypoint_score, player_score, keypoint_stability, homography_stability, player_plausibility):
    # Weight the different components
    weights = {
        'keypoint_score': 0.6,
        'player_score': 0.1,
        'keypoint_stability': 0.1,
        'homography_stability': 0.1,
        'player_plausibility': 0.1
    }
    
    # Scale all scores to 0-100
    keypoint_stability_score = keypoint_stability * 100
    homography_stability_score = homography_stability * 100
    player_plausibility_score = player_plausibility * 100
    
    # Print component scores for debugging
    print(f"\nComponent Scores:")
    print(f"Keypoint Score: {keypoint_score:.2f} (weight: {weights['keypoint_score']})")
    print(f"Player Score: {player_score:.2f} (weight: {weights['player_score']})")
    print(f"Keypoint Stability: {keypoint_stability_score:.2f} (weight: {weights['keypoint_stability']})")
    print(f"Homography Stability: {homography_stability_score:.2f} (weight: {weights['homography_stability']})")
    print(f"Player Plausibility: {player_plausibility_score:.2f} (weight: {weights['player_plausibility']})")
    
    # Calculate weighted score
    final_score = (
        weights['keypoint_score'] * keypoint_score +
        weights['player_score'] * player_score +
        weights['keypoint_stability'] * keypoint_stability_score +
        weights['homography_stability'] * homography_stability_score +
        weights['player_plausibility'] * player_plausibility_score
    )
    
    return final_score

def main():
    parser = argparse.ArgumentParser(description="Process keypoints and estimate homography for each frame.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--video-width', type=int, default=1280, help="Width of the video frame in pixels")
    parser.add_argument('--video-height', type=int, default=720, help="Height of the video frame in pixels")
    args = parser.parse_args()

    results, valid_frames, avg_inlier_ratio, avg_reprojection_error, avg_keypoint_score, player_score, avg_jump, total_jumps, biggest_jump, large_jumps, transitions, avg_keypoint_stability, avg_homography_stability, avg_player_plausibility = process_input_file(args.input, args.video_width, args.video_height)
    
    # Summarize and print score ranges
    score_summary = summarize_scores(results)
    print("\nInlier Ratio Summary:")
    for range_key, count in score_summary['inlier_ratio'].items():
        print(f"{range_key}: {count} frames")
    
    print("\nAverage Reprojection Error Summary:")
    for range_key, count in score_summary['avg_reprojection_error'].items():
        print(f"{range_key}: {count} frames")
    
    print(f"\nTotal frames processed: {len(results)}")
    print(f"Valid frames (with 4 or more keypoints): {valid_frames}")
    print(f"Average Inlier Ratio: {avg_inlier_ratio:.2f}")
    print(f"Average Reprojection Error: {avg_reprojection_error:.2f} pixels")
    print(f"Keypoint Score: {avg_keypoint_score:.2f}")
    
    print(f"\nPlayer Position Consistency Score: {player_score:.2f}")
    print(f"Average Player Jump: {avg_jump:.2f} pixels")
    print(f"Biggest Player Jump: {biggest_jump:.2f} pixels")
    print(f"Total Player Jumps Analyzed: {total_jumps}")
    print(f"Large Jumps (potential substitutions or view changes): {large_jumps}")
    
    print(f"\nScene Transitions Detected: {len(transitions)}")
    print(f"Keypoint Stability Score: {avg_keypoint_stability:.2f}")
    print(f"Homography Stability Score: {avg_homography_stability:.2f}")
    print(f"Player Movement Plausibility Score: {avg_player_plausibility:.2f}")
    
    final_score = calculate_final_score_keypoints(
        avg_keypoint_score, 
        player_score, 
        avg_keypoint_stability, 
        avg_homography_stability, 
        avg_player_plausibility
    )
    print(f"\nFinal Score: {final_score:.2f}")

if __name__ == '__main__':
    main()
