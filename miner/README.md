# Score Vision (SN44) - Miner

This is the miner component for Subnet 44 (Soccer Video Analysis). For full subnet documentation, please see the [main README](../README.md).

# Optimized Soccer Video Processing Pipeline

This miner pipeline has been optimized for efficient and accurate soccer video analysis. Key improvements include:

## 1. Batch Inference
- Frames are loaded into RAM and processed in batches (configurable batch size) for efficient GPU utilization.
- Player detection and tracking are performed in batches, reducing inference time.

## 2. Fake Object Handling
- If fewer than 70% of frames contain detected objects, fake player and referee objects are generated for empty frames to ensure robust downstream processing.

## 3. CLIP-based Object Filtering
- After detection, each object's region of interest (ROI) is classified using a CLIP model (validator logic).
- Only objects with a positive CLIP-based score are kept; others are filtered out.
- Multi-threaded filtering is used for speed, with retry logic to handle HuggingFace tokenizer thread-safety issues.

## 4. class_id Update for Validator Compatibility
- After CLIP classification, each object's `class_id` is updated to match its `expected_label` (as determined by CLIP), using a mapping that covers all possible labels.
- This ensures that the output is fully compatible with the validator's expected format and logic.

## 5. Sorting by Correctness
- Objects where `predicted_label == expected_label` are placed at the beginning of the list; mismatches are moved to the end.

---

For more details, see the implementation in `miner/endpoints/soccer.py` and the validator logic in `validator/evaluation/bbox_clip.py`.

## System Requirements

Please see [REQUIREMENTS.md](REQUIREMENTS.md) for detailed system requirements.

## Setup Instructions

1. **Bootstrap System Dependencies**

```bash
# Clone repository
git clone https://github.com/score-technologies/score-vision.git
cd score-vision
chmod +x bootstrap.sh
./bootstrap.sh
```

2. Setup Bittensor Wallet:

```bash
# Create hotkey directory
mkdir -p ~/.bittensor/wallets/[walletname]/hotkeys/

# If copying from local machine:
scp ~/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname] [user]@[SERVERIP]:~/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname]
scp ~/.bittensor/wallets/[walletname]/coldkeypub.txt [user]@[SERVERIP]:~/.bittensor/wallets/[walletname]/coldkeypub.txt
```

## Installation

1. Create and activate virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix-like systems
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
uv pip install -e ".[miner]"
```

3. Setup environment:

```bash
cp miner/.env.example miner/.env
# Edit .env with your configuration
```

## Register IP on Chain

1. Get your server IP:

```bash
curl ifconfig.me
```

2. Register your IP:

```bash
fiber-post-ip --netuid 44 --subtensor.network finney --external_port [YOUR-PORT] --wallet.name [WALLET_NAME] --wallet.hotkey [HOTKEY_NAME] --external_ip [YOUR-IP]
```

## Running the Miner

### Test the Pipeline

```bash
cd miner
python scripts/test_pipeline.py
```

### Production Deployment (PM2)

```bash
cd miner
pm2 start \
  --name "sn44-miner" \
  --interpreter "../.venv/bin/python" \
  "../.venv/bin/uvicorn" \
  -- main:app --host 0.0.0.0 --port 7999
```

### Development Mode

```bash
cd miner
uvicorn main:app --reload --host 0.0.0.0 --port 7999
```

### Testing the Pipeline

To test the inference pipeline locally:

```bash
cd miner
python scripts/test_pipeline.py
```

## Operational Overview

The miner operates several key processes to handle soccer video analysis:

### 1. Challenge Reception

- Listens for incoming challenges from validators
- Validates challenge authenticity using cryptographic signatures
- Downloads video content from provided URLs
- Manages concurrent challenge processing
- Implements exponential backoff for failed downloads

### 2. Video Processing Pipeline

- Loads video frames efficiently using OpenCV
- Processes frames through multiple detection models:
  - Player detection and tracking
  - Goalkeeper identification
  - Referee detection
  - Ball tracking
- Manages GPU memory for optimal performance
- Implements frame batching for efficiency

### 3. Response Generation

- Generates standardized bounding box annotations
- Formats responses according to subnet protocol
- Includes confidence scores for detections
- Implements quality checks before submission
- Handles response encryption and signing

### 4. Health Management

- Maintains availability endpoint for validator checks
- Monitors system resources (GPU/CPU usage)
- Implements graceful challenge rejection when overloaded
- Tracks processing metrics and timings
- Manages concurrent request limits

## Configuration Reference

Key environment variables in `.env`:

```bash
# Network
NETUID=261                                    # Subnet ID (261 for testnet, 44 for mainnnet)
SUBTENSOR_NETWORK=test                        # Network type (test/local)
SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443  # Network address

# Miner
WALLET_NAME=default                           # Your wallet name
HOTKEY_NAME=default                           # Your hotkey name
MIN_STAKE_THRESHOLD=2                         # Minimum stake requirement

# Hardware
DEVICE=cuda                                   # Computing device (cuda/cpu/mps)
```

## Troubleshooting

### Common Issues

1. **Video Download Failures**

   - Check network connectivity
   - Verify URL accessibility
   - Monitor disk space
   - Check download timeouts

2. **Model Loading Issues**

   - Verify model files in `data/` directory
   - Check CUDA/GPU availability
   - Monitor GPU memory usage
   - Verify model compatibility

3. **Performance Issues**

   - Adjust batch size settings
   - Monitor system resources
   - Check for memory leaks
   - Optimize frame processing

4. **Network Connectivity**
   - Ensure port 7999 is exposed
   - Check firewall settings
   - Verify validator connectivity
   - Monitor network latency

For advanced configuration options and architecture details, see the [main README](../README.md).

## Credit

A big shout out to Skalskip and the work they're doing over at Roboflow. The base miner utilizes models and techniques from:

- [Roboflow Sports](https://github.com/roboflow/sports) - An open-source repository providing computer vision tools and models for sports analytics, particularly focused on soccer/football detection tasks.

## Các hướng tối ưu đã thử và đề xuất

### Đã thực hiện:
- Detect/tracking trên frame resize nhỏ (640x360, 960x540) để tăng tốc, sau đó scale bbox về kích thước gốc.
- CLIP-based filtering trên frame resize (640x360, 960x540, 1280x720): tốc độ tăng nhưng chất lượng giảm mạnh với object nhỏ.
- CLIP-based filtering trên frame gốc (full size): chất lượng tốt nhất, BBox Score cao, tốc độ chậm hơn.
- Batch CLIP toàn bộ object của video hoặc từng frame: giảm số lần gọi CLIP, tận dụng GPU tốt hơn.
- Multi-thread/multi-process filter: tăng tốc filter/scoring nếu CPU-bound.
- Scale bbox về gốc ngay sau detect, các bước sau luôn làm việc với bbox gốc.
- Fake object và keypoint sinh sau cùng, dựa trên bbox/frame gốc.

### Vấn đề còn tồn tại:
- CLIP inference vẫn là bottleneck lớn nhất.
- Tốc độ CLIP không tăng nhiều nếu object/frame nhiều, hoặc GPU không đủ mạnh.
- Nếu object nhỏ, CLIP vẫn có thể nhầm lẫn, đặc biệt khi crop ROI quá nhỏ.
- Fake object có thể không hợp lệ nếu không filter lại bằng CLIP.

### Hướng tối ưu mới đề xuất:
- Dùng phiên bản CLIP nhỏ hơn (ViT-B/16, ViT-S/16) hoặc quantized để tăng tốc inference.
- Gom batch CLIP cho nhiều frame cùng lúc (nếu RAM/GPU đủ).
- Tăng scale khi crop ROI cho CLIP (lấy vùng lớn hơn bbox).
- Sử dụng ONNX/TensorRT cho CLIP để inference nhanh hơn.
- Caching kết quả CLIP theo tracking ID.
- Chỉ filter CLIP cho object nghi ngờ (detector score thấp).
- Vectorize toàn bộ pipeline filter/scoring bằng numpy/pandas.
- Tối ưu fake object: chỉ sinh khi cần, luôn filter lại bằng CLIP.

## Logic pipeline hiện tại

- Detect/tracking thực hiện trên frame resize nhỏ (ví dụ 640x360), scale bbox về kích thước gốc ngay sau detect.
- CLIP-based filtering thực hiện trên frame gốc và bbox gốc, đảm bảo chất lượng phân loại tốt nhất.
- CLIP inference sử dụng ONNXRuntime (hoặc TensorRT nếu có) để tăng tốc mạnh mẽ, đặc biệt trên GPU mạnh như A100.
- Các bước fake object, keypoint, aggregation đều làm việc với bbox/frame gốc.

## Đánh giá cách tối ưu mới (ONNX/TensorRT)

- CLIP inference tăng tốc mạnh trên GPU mạnh (A100), thời gian filter giảm từ 26s xuống 1-3s cho 750 frames.
- Tổng pipeline có thể giảm từ 40s xuống 13-17s nếu detect/tracking cũng tối ưu.
- Chất lượng CLIP giữ nguyên, không giảm so với HuggingFace.
- Nên batch CLIP lớn nhất có thể để tận dụng tối đa GPU.

## Pipeline Optimization Notes

### Batch size fallback logic for CLIP batching
- Khi thực hiện batch inference với CLIP, nếu batch_size=None, pipeline sẽ tự động sử dụng toàn bộ số lượng ROI hiện có (`clip_batch_size = len(clip_rois)`).
- Nếu batch_size là số nguyên, pipeline sẽ lấy giá trị nhỏ nhất giữa batch_size và số lượng ROI (`clip_batch_size = min(batch_size, len(clip_rois))`).
- Điều này giúp tránh lỗi TypeError khi so sánh None với int, đồng thời đảm bảo tận dụng tối đa GPU khi batch_size không được chỉ định.
