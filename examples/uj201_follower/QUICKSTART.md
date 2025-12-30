# Quick Start: UJ201 Camera + VLA Testing

All scripts are located in `examples/uj201_follower/`. Run from the repository root:

## Test Sequence

### 1. Test Cameras Only
```bash
python examples/uj201_follower/test_cameras.py
```
- Validates both cameras work
- Checks frame rates and latency
- Live preview with 'q' to quit, 's' to save

### 2. Test Robot + Cameras
```bash
python examples/uj201_follower/test_robot_cameras.py
```
- Validates robot can read cameras
- Checks observation dict structure
- Displays motor positions + camera feeds

### 3. Test VLA Inference
```bash
python examples/uj201_follower/test_vla_inference.py
```
- Loads SmolVLA model (450M params)
- Runs full control loop: camera → VLA → motors
- Edit `TASK` variable for your instruction

## Camera Setup

| Camera | Index | Resolution | FPS | Purpose |
|--------|-------|------------|-----|---------|
| Wrist  | 0     | 1280x960   | 5   | End-effector view |
| Scene  | 1     | 1920x1080  | 30  | Desktop/external POV |

## Configuration

All test scripts use:
```python
ROBOT_PORT = "/dev/tty.usbmodem5A7A0565221"
CAMERAS = {
    "wrist": OpenCVCameraConfig(camera_index=0, ...),
    "scene": OpenCVCameraConfig(camera_index=1, ...),
}
```

## Expected Results

### Camera Test
- ✅ Both cameras detected and initialized
- ✅ Frames captured successfully
- ✅ Frame rates: wrist ~5fps, scene ~30fps
- ✅ Synchronization: <50ms time difference

### Robot + Camera Test
- ✅ Observation dict contains 10 motor positions + 2 camera frames
- ✅ Total latency: ~250-350ms per observation
- ✅ Achievable rate: 3-4 Hz

### VLA Inference Test
- ✅ Model loads successfully
- ✅ Inference: ~50-150ms per action chunk (on MPS/CUDA)
- ✅ Full cycle: ~300-400ms
- ✅ Control rate: 2-3 Hz

## Troubleshooting

### "Camera not found"
```bash
lerobot-find-cameras opencv
# Should show Camera #0 and Camera #1
```

### "Failed to connect to robot"
```bash
ls /dev/tty.usbmodem*
# Should show /dev/tty.usbmodem5A7A0565221
```

### "Model not found"
```bash
# Download SmolVLA first
huggingface-cli download lerobot/smolvla_base
```

### "Import errors" (cv2, torch, numpy)
These are lint warnings - the packages are installed in the conda environment. Run:
```bash
conda activate UJ201
python examples/uj201_follower/test_cameras.py  # Should work despite lint errors
```

## Output Files

All test scripts save results to timestamped directories:
- `camera_test_results/camera_test_YYYYMMDD_HHMMSS.txt`
- `robot_camera_test_results/robot_camera_test_YYYYMMDD_HHMMSS.txt`
- `vla_inference_results/vla_inference_YYYYMMDD_HHMMSS.txt`

## Safety

⚠️ All scripts use `max_relative_target=10.0` (10° safety limit per step)

⚠️ Motors 2 and 3 are mechanically coupled - see `debug/test_gearbox_shoulder.py` for details

⚠️ Always supervise VLA control during first runs

## Next Steps After Testing

1. ✅ All tests pass → Try VLA with different tasks (edit `TASK` variable)
2. 📊 Collect training data → Use `lerobot-record` (see CAMERA_VLA_README.md)
3. 🎓 Fine-tune SmolVLA → Use `lerobot-train` on your data
4. 🚀 Deploy → Integrate into your application

## Key Files

| File | Purpose |
|------|---------|
| `test_cameras.py` | Camera-only testing |
| `test_robot_cameras.py` | Robot + camera integration |
| `test_vla_inference.py` | Full VLA control loop |
| `collect_data.py` | Data collection for training |
| `CAMERA_VLA_README.md` | Comprehensive documentation |
| `QUICKSTART.md` | This file |
| `debug/test_gearbox_shoulder.py` | Motor coupling tests |

## Example VLA Tasks

Edit the `TASK` variable in `test_vla_inference.py`:

```python
TASK = "pick up the red block and place it in the box"
TASK = "grasp the cup and move it to the left"
TASK = "push the button"
TASK = "open the drawer"
```

SmolVLA will attempt these tasks using visual feedback from both cameras.
