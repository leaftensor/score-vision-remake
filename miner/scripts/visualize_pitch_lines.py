import cv2
import numpy as np
import json
import tempfile
import requests
import time
from pathlib import Path
import sys
import os

# Add validator path to import the functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'validator'))
from validator.evaluation.keypoint_scoring import detect_pitch_lines_tophat

# Test video URL
TEST_VIDEO_URL = "https://scoredata.me/2025_06_18/2025_06_18_d49f45ff/2025_06_18_d49f45ff_195945e20e9e4325b51ab84ff134c7_dcb7b85f.mp4"

def download_video(url):
    """Download video from URL to temporary file"""
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

def draw_pitch_lines(frame, line_segments, color=(0, 255, 255), thickness=2):
    """Draw pitch line segments on frame"""
    overlay = frame.copy()
    
    if line_segments:
        for x1, y1, x2, y2 in line_segments:
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    # Blend with original frame for transparency effect
    alpha = 0.7
    result = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)
    return result

def draw_detection_stages(frame, grass_mask, white_lines, masked, edges, line_mask):
    """Create a multi-panel view showing detection stages"""
    h, w = frame.shape[:2]
    
    # Resize frame and masks to fit in panels
    panel_h, panel_w = h // 2, w // 2
    
    # Create 3x2 grid
    grid = np.zeros((panel_h * 3, panel_w * 2, 3), dtype=np.uint8)
    
    # Original frame
    original_resized = cv2.resize(frame, (panel_w, panel_h))
    grid[0:panel_h, 0:panel_w] = original_resized
    
    # Grass mask
    grass_colored = cv2.applyColorMap(cv2.resize(grass_mask, (panel_w, panel_h)), cv2.COLORMAP_JET)
    grid[0:panel_h, panel_w:panel_w*2] = grass_colored
    
    # White lines
    white_colored = cv2.applyColorMap(cv2.resize(white_lines, (panel_w, panel_h)), cv2.COLORMAP_HOT)
    grid[panel_h:panel_h*2, 0:panel_w] = white_colored
    
    # Masked (white lines on grass)
    masked_colored = cv2.applyColorMap(cv2.resize(masked, (panel_w, panel_h)), cv2.COLORMAP_COOL)
    grid[panel_h:panel_h*2, panel_w:panel_w*2] = masked_colored
    
    # Edges
    edges_colored = cv2.applyColorMap(cv2.resize(edges, (panel_w, panel_h)), cv2.COLORMAP_PLASMA)
    grid[panel_h*2:panel_h*3, 0:panel_w] = edges_colored
    
    # Final line mask
    line_colored = cv2.applyColorMap(cv2.resize(line_mask, (panel_w, panel_h)), cv2.COLORMAP_VIRIDIS)
    grid[panel_h*2:panel_h*3, panel_w:panel_w*2] = line_colored
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    cv2.putText(grid, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Grass Mask", (panel_w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, "White Lines", (10, panel_h + 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Masked Lines", (panel_w + 10, panel_h + 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Edges", (10, panel_h*2 + 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Final Lines", (panel_w + 10, panel_h*2 + 30), font, font_scale, color, thickness)
    
    return grid

def visualize_pitch_detection(video_path, output_path="pitch_detection_visualization.mp4", 
                            show_stages=False, max_frames=300):
    """
    Visualize pitch line detection on video
    
    Args:
        video_path: Path to input video
        output_path: Path for output video
        show_stages: If True, show detection process stages
        max_frames: Maximum number of frames to process
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    if show_stages:
        # For stages view, output is 3x2 grid of half-size panels
        out_width, out_height = width, height // 2 * 3
    else:
        out_width, out_height = width, height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    frame_count = 0
    process_frames = min(max_frames, total_frames)
    
    print(f"Processing {process_frames} frames...")
    
    # Statistics tracking
    total_lines_detected = 0
    frames_with_lines = 0
    
    while frame_count < process_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run pitch line detection
        try:
            grass, white_lines, masked, edges, line_mask, final_kept = detect_pitch_lines_tophat(frame)
            
            # Update statistics
            if final_kept:
                total_lines_detected += len(final_kept)
                frames_with_lines += 1
            
            if show_stages:
                # Create multi-panel visualization
                output_frame = draw_detection_stages(frame, grass, white_lines, masked, edges, line_mask)
            else:
                # Draw detected lines on original frame
                output_frame = draw_pitch_lines(frame, final_kept)
                
                # Add info overlay
                info_text = f"Frame {frame_count + 1}/{process_frames} | Lines: {len(final_kept) if final_kept else 0}"
                cv2.putText(output_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2)
            
            out.write(output_frame)
            
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # Write original frame if detection fails
            if show_stages:
                output_frame = np.zeros((height // 2 * 3, width, 3), dtype=np.uint8)
                output_frame[0:height//2, 0:width] = cv2.resize(frame, (width, height//2))
            else:
                output_frame = frame
            out.write(output_frame)
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 50 == 0:
            progress = (frame_count / process_frames) * 100
            avg_lines = total_lines_detected / frames_with_lines if frames_with_lines > 0 else 0
            print(f"Progress: {progress:.1f}% | Avg lines per frame: {avg_lines:.1f}")
    
    cap.release()
    out.release()
    
    # Final statistics
    avg_lines = total_lines_detected / frames_with_lines if frames_with_lines > 0 else 0
    detection_rate = (frames_with_lines / frame_count) * 100 if frame_count > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"PITCH LINE DETECTION STATISTICS")
    print(f"{'='*50}")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with lines detected: {frames_with_lines}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Total lines detected: {total_lines_detected}")
    print(f"Average lines per frame: {avg_lines:.1f}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*50}")

def main():
    print("🏟️  Pitch Line Detection Visualization")
    print("=" * 50)
    
    # Download video
    print("Downloading test video...")
    video_path = download_video(TEST_VIDEO_URL)
    
    try:
        # Create two visualizations
        print("\n1. Creating main visualization with detected lines...")
        visualize_pitch_detection(
            video_path, 
            "pitch_lines_main.mp4", 
            show_stages=False,
            max_frames=150  # Process first 150 frames
        )
        
        print("\n2. Creating detailed visualization showing detection stages...")
        visualize_pitch_detection(
            video_path, 
            "pitch_lines_stages.mp4", 
            show_stages=True,
            max_frames=50  # Fewer frames for detailed view
        )
        
        print("\n✅ Visualization complete!")
        print("Generated files:")
        print("  - pitch_lines_main.mp4: Main video with detected lines overlay")
        print("  - pitch_lines_stages.mp4: Detailed view of detection process")
        
    finally:
        # Cleanup
        try:
            os.unlink(video_path)
            print(f"\n🧹 Cleaned up temporary video file")
        except:
            pass

if __name__ == "__main__":
    main() 