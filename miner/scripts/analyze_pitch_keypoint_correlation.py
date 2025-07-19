import cv2
import numpy as np
import json
import tempfile
import requests
import time
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# Add validator path to import the functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'validator'))
from validator.evaluation.keypoint_scoring import (
    detect_pitch_lines_tophat, 
    mean_keypoint_to_line_distance_score,
    point_to_segment_dist,
    get_valid_keypoints
)

# Test video URL and keypoint results file
TEST_VIDEO_URL = "https://scoredata.me/2025_06_18/2025_06_18_d49f45ff/2025_06_18_d49f45ff_195945e20e9e4325b51ab84ff134c7_dcb7b85f.mp4"
INPUT_KEYPOINTS_FILE = r"C:\Users\longp\Documents\GitHub\score-vision\miner\test_outputs\pipeline_results_1752834642.json"

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

def analyze_keypoint_line_correlation(frame, keypoints, line_segments, video_width, video_height):
    """
    Analyze correlation between keypoints and detected lines
    
    Returns:
        dict: Analysis results including distances, scores, etc.
    """
    results = {
        'keypoints': [],
        'line_segments': line_segments,
        'distances': [],
        'nearest_lines': [],
        'score': 0.0,
        'valid_keypoints_count': 0
    }
    
    if not keypoints or not line_segments:
        return results
    
    # Get valid keypoints
    keypoints_array = np.array(keypoints).reshape(-1, 2)
    valid_kps, valid_indices = get_valid_keypoints(keypoints_array, video_width, video_height)
    
    if len(valid_kps) == 0:
        return results
    
    results['valid_keypoints_count'] = len(valid_kps)
    results['keypoints'] = valid_kps.tolist()
    
    # Normalize coordinates
    norm_kps = valid_kps / np.array([video_width, video_height])
    norm_segs = [(
        x1 / video_width, y1 / video_height,
        x2 / video_width, y2 / video_height
    ) for (x1, y1, x2, y2) in line_segments]
    
    # Calculate distances to nearest lines
    for i, (u, v) in enumerate(norm_kps):
        distances_to_lines = []
        for j, (x1, y1, x2, y2) in enumerate(norm_segs):
            dist = point_to_segment_dist(u, v, x1, y1, x2, y2)
            distances_to_lines.append((dist, j))
        
        # Find nearest line
        min_dist, nearest_line_idx = min(distances_to_lines)
        results['distances'].append(min_dist)
        results['nearest_lines'].append(nearest_line_idx)
    
    # Calculate score
    results['score'] = mean_keypoint_to_line_distance_score(
        keypoints_array, line_segments, video_width, video_height
    )
    
    return results

def create_correlation_visualization(frame, analysis, video_width, video_height, 
                                   frame_idx=0, save_path=None):
    """
    Create a detailed visualization showing keypoint-line correlation
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original frame with lines and keypoints
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Frame {frame_idx}: Detected Lines & Keypoints')
    
    # Draw lines
    for line in analysis['line_segments']:
        x1, y1, x2, y2 = line
        ax1.plot([x1, x2], [y1, y2], 'yellow', linewidth=1, alpha=0.8)
    
    # Draw keypoints
    for i, (kx, ky) in enumerate(analysis['keypoints']):
        ax1.plot(kx, ky, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax1.text(kx + 20, ky - 20, f'K{i}', color='white', fontweight='bold', fontsize=10)
    
    ax1.set_xlim(0, video_width)
    ax1.set_ylim(video_height, 0)
    ax1.axis('off')
    
    # 2. Distance heatmap
    if analysis['distances']:
        ax2.hist(analysis['distances'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(analysis['distances']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(analysis["distances"]):.4f}')
        ax2.set_title('Distance Distribution (Normalized)')
        ax2.set_xlabel('Distance to Nearest Line')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Keypoint-Line connections
    ax3.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax3.set_title(f'Keypoint-Line Connections (Score: {analysis["score"]:.3f})')
    
    # Draw lines in light gray
    for line in analysis['line_segments']:
        x1, y1, x2, y2 = line
        ax3.plot([x1, x2], [y1, y2], 'lightgray', linewidth=1, alpha=0.5)
    
    # Draw connections from keypoints to nearest lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(analysis['keypoints'])))
    for i, (kx, ky) in enumerate(analysis['keypoints']):
        if i < len(analysis['nearest_lines']):
            nearest_line_idx = analysis['nearest_lines'][i]
            if nearest_line_idx < len(analysis['line_segments']):
                line = analysis['line_segments'][nearest_line_idx]
                lx1, ly1, lx2, ly2 = line
                
                # Find closest point on line segment
                norm_kx, norm_ky = kx / video_width, ky / video_height
                norm_lx1, norm_ly1 = lx1 / video_width, ly1 / video_height
                norm_lx2, norm_ly2 = lx2 / video_width, ly2 / video_height
                
                # Calculate projection
                vx, vy = norm_lx2 - norm_lx1, norm_ly2 - norm_ly1
                wx, wy = norm_kx - norm_lx1, norm_ky - norm_ly1
                t = (wx*vx + wy*vy) / (vx*vx + vy*vy) if (vx*vx + vy*vy) > 0 else 0
                t = max(0, min(1, t))
                
                closest_x = (norm_lx1 + t*vx) * video_width
                closest_y = (norm_ly1 + t*vy) * video_height
                
                # Draw connection
                ax3.plot([kx, closest_x], [ky, closest_y], color=colors[i], 
                        linewidth=2, alpha=0.8)
                ax3.plot(kx, ky, 'o', color=colors[i], markersize=8, 
                        markeredgecolor='white', markeredgewidth=2)
                ax3.plot(closest_x, closest_y, 's', color=colors[i], markersize=6)
    
    ax3.set_xlim(0, video_width)
    ax3.set_ylim(video_height, 0)
    ax3.axis('off')
    
    # 4. Score breakdown
    if analysis['distances']:
        mean_dist = np.mean(analysis['distances'])
        k = 0.0037
        x0 = 0.02
        
        # Plot logistic function
        x_range = np.linspace(0, 0.1, 1000)
        y_range = 1.0 / (1.0 + np.exp((x_range - x0) / k))
        
        ax4.plot(x_range, y_range, 'b-', linewidth=2, label='Logistic Function')
        ax4.axvline(mean_dist, color='red', linestyle='--', 
                   label=f'Current Mean Dist: {mean_dist:.4f}')
        ax4.axhline(analysis['score'], color='green', linestyle='--',
                   label=f'Current Score: {analysis["score"]:.3f}')
        ax4.axvline(x0, color='orange', linestyle=':', alpha=0.7,
                   label=f'Threshold x0: {x0}')
        
        ax4.set_title('Scoring Function')
        ax4.set_xlabel('Normalized Distance')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 0.1)
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_video_frames(video_path, keypoints_data, max_frames=10):
    """
    Analyze correlation across multiple video frames
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Analyzing video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    frames_data = keypoints_data.get('frames', keypoints_data) if 'frames' in keypoints_data else keypoints_data
    
    analyses = []
    frame_indices = sorted([int(k) for k in frames_data.keys()])[:max_frames]
    
    print(f"Processing {len(frame_indices)} frames with keypoint data...")
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Get keypoints for this frame
        frame_data = frames_data[str(frame_idx)]
        keypoints = frame_data.get('keypoints', [])
        
        # Detect pitch lines
        try:
            _, _, _, _, _, line_segments = detect_pitch_lines_tophat(frame)
            
            # Analyze correlation
            analysis = analyze_keypoint_line_correlation(
                frame, keypoints, line_segments, width, height
            )
            analysis['frame_idx'] = frame_idx
            analyses.append(analysis)
            
            # Create visualization for selected frames
            if i < 5:  # Save first 5 detailed visualizations
                save_path = f"keypoint_line_analysis_frame_{frame_idx}.png"
                create_correlation_visualization(
                    frame, analysis, width, height, frame_idx, save_path
                )
                print(f"Saved analysis for frame {frame_idx}: Score = {analysis['score']:.3f}")
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
    
    cap.release()
    return analyses

def create_summary_plots(analyses):
    """
    Create summary plots across all analyzed frames
    """
    if not analyses:
        print("No analyses to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    frame_indices = [a['frame_idx'] for a in analyses]
    scores = [a['score'] for a in analyses]
    line_counts = [len(a['line_segments']) for a in analyses]
    keypoint_counts = [a['valid_keypoints_count'] for a in analyses]
    mean_distances = [np.mean(a['distances']) if a['distances'] else 0 for a in analyses]
    
    # 1. Score evolution
    ax1.plot(frame_indices, scores, 'bo-', linewidth=2, markersize=6)
    ax1.set_title('Keypoint-Line Alignment Score by Frame')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Alignment Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Detection counts
    ax2.bar(range(len(frame_indices)), line_counts, alpha=0.7, label='Lines Detected', color='skyblue')
    ax2_twin = ax2.twinx()
    ax2_twin.bar([x + 0.4 for x in range(len(frame_indices))], keypoint_counts, 
                alpha=0.7, label='Valid Keypoints', color='orange', width=0.4)
    
    ax2.set_title('Detection Counts by Frame')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Lines Detected', color='blue')
    ax2_twin.set_ylabel('Valid Keypoints', color='orange')
    ax2.set_xticks(range(len(frame_indices)))
    ax2.set_xticklabels([str(idx) for idx in frame_indices], rotation=45)
    
    # 3. Distance distribution
    all_distances = []
    for a in analyses:
        all_distances.extend(a['distances'])
    
    if all_distances:
        ax3.hist(all_distances, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(np.mean(all_distances), color='red', linestyle='--', 
                   label=f'Overall Mean: {np.mean(all_distances):.4f}')
        ax3.set_title('Overall Distance Distribution')
        ax3.set_xlabel('Normalized Distance to Nearest Line')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Score vs. Mean Distance correlation
    if mean_distances and scores:
        ax4.scatter(mean_distances, scores, c=frame_indices, cmap='viridis', s=60, alpha=0.7)
        ax4.set_title('Score vs. Mean Distance Correlation')
        ax4.set_xlabel('Mean Distance to Lines')
        ax4.set_ylabel('Alignment Score')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Frame Index')
    
    plt.tight_layout()
    plt.savefig('keypoint_line_summary_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("🔍 Keypoint-Line Correlation Analysis")
    print("=" * 50)
    
    # Load keypoint data
    print("Loading keypoint data...")
    with open(INPUT_KEYPOINTS_FILE, 'r') as f:
        keypoints_data = json.load(f)
    
    print(f"Found keypoint data for {len(keypoints_data.get('frames', keypoints_data))} frames")
    
    # Download video
    print("Downloading test video...")
    video_path = download_video(TEST_VIDEO_URL)
    
    try:
        # Analyze correlation
        print("\nAnalyzing keypoint-line correlation...")
        analyses = analyze_video_frames(video_path, keypoints_data, max_frames=10)
        
        print(f"\nAnalyzed {len(analyses)} frames")
        
        if analyses:
            # Print summary statistics
            scores = [a['score'] for a in analyses]
            line_counts = [len(a['line_segments']) for a in analyses]
            keypoint_counts = [a['valid_keypoints_count'] for a in analyses]
            
            print(f"\n📊 SUMMARY STATISTICS")
            print(f"=" * 30)
            print(f"Average alignment score: {np.mean(scores):.3f}")
            print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"Average lines detected: {np.mean(line_counts):.1f}")
            print(f"Average valid keypoints: {np.mean(keypoint_counts):.1f}")
            
            # Create summary plots
            print("\nCreating summary visualizations...")
            create_summary_plots(analyses)
            
            print("\n✅ Analysis complete!")
            print("Generated files:")
            print("  - keypoint_line_analysis_frame_*.png: Detailed frame analyses")
            print("  - keypoint_line_summary_analysis.png: Overall summary")
        else:
            print("❌ No frames could be analyzed")
            
    finally:
        # Cleanup
        try:
            os.unlink(video_path)
            print(f"\n🧹 Cleaned up temporary video file")
        except:
            pass

if __name__ == "__main__":
    main() 