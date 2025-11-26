# UJ201 Camera + VLA Integration

Integration of dual cameras with UJ201 robot arm for vision-language-action (VLA) control using SmolVLA.

## Hardware Setup

### Cameras
- **Camera 0 (Wrist)**: 1280x960 @ 5fps - mounted on robot wrist
- **Camera 1 (Scene)**: 1920x1080 @ 30fps - desktop/external viewpoint

Detected via:
```bash
lerobot-find-cameras opencv
```

### Robot
- **UJ201 Follower**: 9-motor arm with Feetech STS3215 servos
- **Port**: `/dev/tty.usbmodem5A7A0565221`
- **Motors**: shoulder (ID 3), gearbox (ID 2), universal_joint (ID 1), + 6 SO101 motors

## Test Scripts

### 1. Camera Testing (`test_uj201_cameras.py`)

Tests cameras independently without robot:

```bash
python test_uj201_cameras.py
```

**Features:**
- ✅ Camera detection and initialization
- ✅ Single frame capture validation
- ✅ Frame rate analysis (actual vs expected FPS)
- ✅ Synchronized dual capture testing
- ✅ Live side-by-side display (press 'q' to quit, 's' to save)
- ✅ Performance benchmarking

**Output:**
- Test log: `camera_test_results/camera_test_YYYYMMDD_HHMMSS.txt`
- Sample frames: `camera_test_results/wrist_test_*.jpg`, `scene_test_*.jpg`
- Snapshots: `camera_test_results/snapshot_*.jpg`

### 2. Robot + Camera Integration (`test_uj201_with_cameras.py`)

Tests robot with cameras integrated:

```bash
python test_uj201_with_cameras.py
```

**Features:**
- ✅ Robot initialization with camera configs
- ✅ `get_observation()` returns motor positions + camera frames
- ✅ Observation structure validation for VLA compatibility
- ✅ Latency benchmarking (observation capture speed)
- ✅ Live display with motor position overlay
- ✅ Control loop frequency analysis

**Output:**
- Test log: `robot_camera_test_results/robot_camera_test_YYYYMMDD_HHMMSS.txt`
- Sample frames with motor data
- Observation snapshots

**Expected Observation Dict:**
```python
{
    "shoulder.pos": 36.53,      # degrees
    "gearbox.pos": -34.68,      # degrees
    "universal_joint.pos": ...,
    # ... 7 more motor positions
    "wrist": np.array(...),     # (960, 1280, 3) uint8
    "scene": np.array(...),     # (1080, 1920, 3) uint8
}
```

### 3. VLA Inference (`test_uj201_vla_inference.py`)

Full SmolVLA control loop:

```bash
python test_uj201_vla_inference.py
```

**Features:**
- ✅ SmolVLA model loading (`lerobot/smolvla_base`, 450M params)
- ✅ Preprocessor/postprocessor pipeline setup
- ✅ Device selection (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
- ✅ Complete inference loop: observation → preprocessing → VLA → postprocessing → action
- ✅ Action chunk execution (50 steps per inference)
- ✅ Performance metrics (inference time, action send time, achievable Hz)

**Configuration:**
Edit these in the script:
- `TASK`: Natural language instruction (e.g., "pick up the red block and place it in the box")
- `ROBOT_TYPE`: "uj201_follower"
- `MAX_EPISODES`: Number of episodes to run (default: 3)
- `MAX_STEPS_PER_EPISODE`: Steps per episode (default: 50)

**Output:**
- Test log: `vla_inference_results/vla_inference_YYYYMMDD_HHMMSS.txt`
- Performance statistics

## Robot Configuration

The `UJ201Follower` class already supports cameras via config:

```python
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig

cameras = {
    "wrist": OpenCVCameraConfig(
        name="wrist",
        camera_index=0,
        width=1280,
        height=960,
        fps=5,
    ),
    "scene": OpenCVCameraConfig(
        name="scene",
        camera_index=1,
        width=1920,
        height=1080,
        fps=30,
    ),
}

robot_config = UJ201FollowerConfig(
    port="/dev/tty.usbmodem5A7A0565221",
    cameras=cameras,
    use_degrees=True,
    max_relative_target=10.0,  # Safety: max 10° change per step
)

robot = UJ201Follower(robot_config)
robot.connect()

# Get observation with cameras
obs = robot.get_observation()
# obs contains: motor positions + camera frames
```

## VLA Model Options

### SmolVLA (Recommended)
- **Size**: 450M parameters
- **Best for**: Fine-tuning on custom data, efficient inference
- **Multi-camera**: ✅ Yes (automatically resizes to 512x512)
- **Language**: ✅ Yes (natural language task specification)
- **Action chunks**: 50 steps
- **Model ID**: `lerobot/smolvla_base`

### Pi0
- **Size**: 3B parameters
- **Best for**: Zero-shot generalization, cross-embodiment transfer
- **Multi-camera**: ✅ Yes
- **Language**: ✅ Yes
- **GPU**: Requires more VRAM than SmolVLA
- **Model ID**: `lerobot/pi0`

### Pi05
- **Size**: 3B parameters
- **Best for**: Enhanced open-world generalization
- **Enhancement**: Improved over Pi0
- **Model ID**: `lerobot/pi05`

### ACT (Baseline)
- **Size**: 80M parameters
- **Best for**: Lightweight testing, no language needed
- **Multi-camera**: ✅ Yes
- **Language**: ❌ No (vision-only)
- **Model ID**: `lerobot/act`

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA Control Loop                          │
└─────────────────────────────────────────────────────────────┘

1. Robot.get_observation()
   ↓
   {motors: {pos1, pos2, ...}, cameras: {wrist, scene}}

2. build_inference_frame(obs, task, robot_type)
   ↓
   Formatted frame with task string

3. Preprocessor
   ↓
   - Tokenize language (task)
   - Normalize motor positions
   - Resize images to 512x512
   - Convert to tensors
   - Transfer to device (GPU/CPU)

4. SmolVLA.select_action(obs)
   ↓
   Action chunk (50 steps)

5. Postprocessor
   ↓
   - Unnormalize actions
   - Convert from tensors
   - Clip to safe ranges

6. Robot.send_action(action)
   ↓
   Motors execute action

7. Repeat for next step
```

## Next Steps

### Option 1: Test with Pretrained Model
Run `test_uj201_vla_inference.py` to test SmolVLA zero-shot performance on your task.

### Option 2: Collect Training Data
Use `lerobot-record` to collect demonstration data:

```bash
lerobot-record \
  --robot-path lerobot.robots.uj201_follower \
  --robot-overrides '{port: /dev/tty.usbmodem5A7A0565221, cameras: {...}}' \
  --fps 30 \
  --repo-id ${HF_USER}/uj201_demos \
  --num-episodes 50 \
  --warmup-time-s 5 \
  --episode-time-s 30
```

### Option 3: Fine-tune SmolVLA
Train on your collected data:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/uj201_demos \
  --training.offline_steps=10000 \
  --training.eval_freq=1000 \
  --training.save_freq=1000 \
  --output_dir=outputs/smolvla_uj201
```

### Option 4: Deploy for Real Tasks
Integrate VLA control into your application:

```python
from test_uj201_vla_inference import *

# Initialize once
model = SmolVLAPolicy.from_pretrained("your-fine-tuned-model")
robot.connect()

# Control loop
while True:
    obs = robot.get_observation()
    action = model.predict(obs, task="your task")
    robot.send_action(action)
```

## Performance Expectations

### Camera Latency
- **Wrist camera**: ~200ms per frame (5 FPS)
- **Scene camera**: ~33ms per frame (30 FPS)
- **Dual capture**: ~250ms (sequential capture)

### VLA Inference (SmolVLA on Apple M1/M2 MPS)
- **Inference**: ~50-150ms per action chunk
- **Full cycle**: ~300-400ms (observation + inference + action)
- **Achievable control rate**: 2-3 Hz

### VLA Inference (SmolVLA on NVIDIA GPU)
- **Inference**: ~20-50ms per action chunk
- **Full cycle**: ~100-200ms
- **Achievable control rate**: 5-10 Hz

## Troubleshooting

### Camera not detected
```bash
# List available cameras
lerobot-find-cameras opencv

# Check camera permissions (macOS)
# System Settings → Privacy & Security → Camera
```

### Motor communication errors
```bash
# Verify port
ls /dev/tty.usbmodem*

# Check calibration files
ls ~/.cache/lerobot/calibration/
```

### Model download issues
```bash
# Pre-download model
huggingface-cli download lerobot/smolvla_base

# Check available models
huggingface-cli scan-cache
```

### Low FPS / High latency
- Use CUDA/MPS instead of CPU
- Reduce camera resolution
- Decrease `MAX_STEPS_PER_EPISODE`
- Use action chunking (model outputs multiple steps per inference)

## Dependencies

Already installed in lerobot environment:
- `torch` (PyTorch with MPS/CUDA support)
- `opencv-python` (cv2)
- `numpy`
- `transformers` (for SmolVLA)
- `lerobot` (robot control + VLA policies)

## Safety Notes

⚠️ **Safety Limits**:
- `max_relative_target=10.0` limits motor movement to 10° per step
- SmolVLA outputs are clipped to safe ranges by postprocessor
- Always supervise first VLA runs
- Keep emergency stop accessible

⚠️ **Mechanical Coupling**:
- Motors 2 (gearbox) and 3 (shoulder) are mechanically coupled at 2:1 ratio
- VLA must learn this constraint during training
- See `test_gearbox_shoulder.py` for coupling details

## References

- **SmolVLA Paper**: [HuggingFace SmolVLM](https://huggingface.co/blog/smolvlm)
- **LeRobot Docs**: [lerobot.huggingface.co](https://lerobot.huggingface.co)
- **Pi0 Paper**: [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi0)
- **UJ201 Robot**: Custom 9-motor arm based on SO101 design
