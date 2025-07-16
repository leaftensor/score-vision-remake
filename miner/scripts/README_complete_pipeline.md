# Complete Pipeline Script

## Overview

The `complete_pipeline.py` script combines video processing, detection, evaluation, and scoring into a single comprehensive workflow. It merges the functionality of `test_pipeline.py` and `score_detection.py` into one streamlined process.

## Features

### 🔄 **Complete Workflow**
- **Video Download**: Automatically downloads test video from URL
- **Model Management**: Downloads and loads required AI models
- **Detection Processing**: Runs soccer video analysis with keypoint and bbox detection
- **Result Optimization**: Optimizes detection results for storage and transmission
- **Video Annotation**: Creates annotated video with bounding boxes and keypoints
- **Evaluation**: Uses validator to score detection results
- **Comprehensive Scoring**: Calculates final scores with detailed metrics
- **Result Storage**: Saves all outputs to organized files

### 📊 **Scoring System**
- **BBox Score**: Evaluates bounding box detection accuracy
- **Keypoint Score**: Evaluates keypoint detection precision
- **Speed Score**: Calculates processing speed using validator's exponential scaling
- **Final Score**: Combined score (50% bbox + 50% keypoint)
- **FPS Analysis**: Alternative speed scoring based on frames per second

### 🎯 **Output Files**
1. **Detection Results**: JSON file with optimized detection data
2. **Annotated Video**: MP4 file with visual annotations
3. **Comprehensive Results**: JSON file with all scoring metrics
4. **Console Output**: Detailed real-time progress and final summary

## Usage

```bash
cd miner/scripts
python complete_pipeline.py
```

## Pipeline Steps

### Step 1: Model Preparation
- Downloads required AI models (YOLO, CLIP, etc.)
- Checks model availability and integrity

### Step 2: Video Download
- Downloads test video from configured URL
- Handles download progress and error checking

### Step 3: Model Initialization
- Detects optimal device (CPU/GPU)
- Loads all required models into memory
- Initializes model manager

### Step 4: Video Processing
- Processes soccer video for detection
- Extracts keypoints and bounding boxes
- Applies object filtering (hybrid GPU filter)
- Records processing time

### Step 5: Result Optimization
- Optimizes coordinate precision (2 decimal places)
- Filters invalid keypoints (0,0 coordinates)
- Compresses data for efficient storage

### Step 6: Detection Results Storage
- Saves optimized detection results to JSON
- Records data size and processing metrics

### Step 7: Video Annotation
- Creates annotated video with visual overlays
- Draws bounding boxes with class colors
- Marks keypoints on frames
- Saves as MP4 file

### Step 8: Detection Evaluation
- Uses validator to evaluate detection quality
- Calculates bbox and keypoint scores
- Applies scoring algorithms

### Step 9: Score Calculation
- Calculates final combined score
- Computes speed scores using multiple methods
- Generates detailed performance metrics

### Step 10: Results Storage
- Saves comprehensive results to JSON
- Provides detailed console output
- Cleans up temporary files

## Configuration

### Video URL
```python
TEST_VIDEO_URL = "https://scoredata.me/2025_06_18/2025_06_18_d49f45ff/2025_06_18_d49f45ff_195945e20e9e4325b51ab84ff134c7_dcb7b85f.mp4"
```

### Processing Time Limit
```python
MAX_PROCESSING_TIME = 15.0  # seconds
```

## Output Structure

### Detection Results File
```json
{
  "frames": {
    "0": {
      "objects": [
        {
          "bbox": [x1, y1, x2, y2],
          "class_id": 1,
          "confidence": 0.95
        }
      ],
      "keypoints": [[x, y], [x, y], ...]
    }
  },
  "processing_time": 12.34
}
```

### Comprehensive Results File
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "video_path": "/path/to/video.mp4",
  "processing_time": 12.34,
  "total_frames": 300,
  "fps": 24.32,
  "bbox_score": 0.8542,
  "keypoint_score": 0.9234,
  "final_score": 0.8888,
  "speed_score_validator": 0.9876,
  "speed_score_fps": 85.67,
  "frame_scores": {...},
  "feedback": {...}
}
```

## Performance Metrics

### Speed Scoring
- **Validator Method**: Exponential scaling based on processing time
- **FPS Method**: Linear scoring based on frames per second
- **Target FPS**: 30 FPS = 100 score, 15 FPS = 50 score

### Quality Scoring
- **BBox Score**: Evaluates object detection accuracy
- **Keypoint Score**: Evaluates keypoint detection precision
- **Final Score**: Weighted combination (50% each)

## Error Handling

- **Model Download**: Automatic retry and fallback
- **Video Download**: Progress tracking and error recovery
- **Processing**: Graceful handling of detection failures
- **Cleanup**: Automatic temporary file removal
- **Memory Management**: Model cache clearing

## Dependencies

### Required Modules
- `asyncio`: Asynchronous processing
- `cv2`: Video processing and annotation
- `numpy`: Numerical operations
- `loguru`: Logging and progress tracking
- `requests`: Video downloading
- `pathlib`: File path management

### Miner Modules
- `utils.model_manager`: Model loading and management
- `utils.video_downloader`: Video download functionality
- `endpoints.soccer`: Soccer video processing
- `utils.device`: Device detection and optimization
- `scripts.download_models`: Model download utilities

### Validator Modules
- `validator.evaluation.evaluation`: Scoring and evaluation
- `validator.challenge.challenge_types`: Data structures
- `validator.evaluation.evaluation`: Color definitions

## Benefits

### 🚀 **Streamlined Workflow**
- Single command execution
- Automatic progression through all steps
- Comprehensive error handling

### 📈 **Performance Monitoring**
- Real-time progress tracking
- Detailed timing breakdown
- Multiple scoring methodologies

### 🎯 **Quality Assurance**
- Validator-based evaluation
- Multiple output formats
- Comprehensive result storage

### 🔧 **Maintainability**
- Modular function design
- Clear step separation
- Extensive documentation

## Troubleshooting

### Common Issues
1. **Model Download Failures**: Check internet connection and disk space
2. **GPU Memory Errors**: Reduce batch size or use CPU
3. **Video Download Issues**: Verify URL accessibility
4. **Processing Timeouts**: Adjust MAX_PROCESSING_TIME

### Performance Optimization
- Use GPU for faster processing
- Ensure sufficient RAM (8GB+ recommended)
- Use SSD for faster I/O operations
- Close other applications during processing

## Future Enhancements

- **Batch Processing**: Process multiple videos simultaneously
- **Custom Video Support**: Accept local video files
- **Real-time Processing**: Stream processing capabilities
- **Advanced Filtering**: More sophisticated object filtering
- **Performance Profiling**: Detailed performance analysis
- **Web Interface**: GUI for easier interaction 

## Phân tích tối ưu batch_evaluate_frame_filter

### Hiện trạng code (đã khá tối ưu):

#### 1. ROI Gathering
- ✅ Vectorized operations: Chuyển đổi BGR→RGB batch, xử lý nhiều bbox cùng lúc
- ✅ Smart filtering: Loại bỏ bbox nhỏ, scoreboard zone, bounds checking
- ✅ Importance scoring: `calculate_bbox_importance()` để ưu tiên object quan trọng
- ✅ Smart reduction: Giữ 65% object có thể giảm, luôn giữ football và tracked people

#### 2. CLIP Classification
- ✅ Heuristic pre-filtering: Dùng color analysis để loại grass/sky/background mà không cần CLIP
- ✅ Cached text features: `get_cached_text_features()` để tránh encode text lại
- ✅ Batch processing: Xử lý theo batch_size (2048)
- ✅ Memory optimization: Cleanup GPU cache, autocast

#### 3. Scoring & Assignment
- ✅ Batch scoring: Tính điểm cho tất cả object cùng lúc
- ✅ Efficient mapping: Map kết quả về từng frame

---

### Các điểm có thể tối ưu thêm:

#### 1. Tối ưu ROI Extraction
- Dùng vectorized numpy thay vì for-loop cho lọc bbox hợp lệ.

#### 2. Tối ưu CLIP Inference
- Gom toàn bộ ROI cần CLIP vào một batch lớn nhất có thể (tùy GPU RAM), ưu tiên dùng TensorRT nếu có.

#### 3. Tối ưu Memory Management
- Dùng autocast, gradient checkpointing, cleanup cache hợp lý.

#### 4. Tối ưu Heuristic Pre-filtering
- Batch color analysis bằng numpy thay vì từng ROI.

#### 5. Tối ưu Scoring
- Vectorized scoring thay vì for-loop từng object.

#### 6. Parallel Processing
- Dùng ThreadPoolExecutor cho ROI extraction nếu I/O bound.

---

### Tổng kết

- Code hiện tại đã rất tối ưu với smart heuristic, batch processing, memory management, importance-based reduction.
- Tối ưu thêm chủ yếu là về vectorization, TensorRT, parallelism, và memory efficiency.
- Nếu cần tối ưu sâu hơn, cân nhắc tích hợp TensorRT cho image encoder, vectorized numpy cho mọi bước, và xử lý song song. 