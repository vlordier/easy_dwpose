# easy_dwpose

Recently, I tried to inference the DWPose (improved OpenPose) preprocessor for [Diffusers](https://github.com/huggingface/diffusers) and was shocked by how complicated it actually is!
So, I decided to change that!

The goal of Easy DWPose is to provide a generic, reliable, and easy-to-use interface for making skeletons for ControlNet.

Me: <a href="https://x.com/igorfeelippov"><img alt="X account" src="https://img.shields.io/twitter/url/https/twitter.com/diffuserslib.svg?style=social&label=Follow%20%40igorfeelippov"></a>

## Why you should use it :yum:

1. Easy installation!
2. Automatic checkpoint downloading.
3. Generic class to either import in Jupyter or to run on a video/images.
4. Code that is easy to read and modify.
5. Choose GPU for multi-gpu inference!
6. **MPS support for Apple Silicon devices!**
7. Custom drawing functions: convenient interface for modifying *how* you draw skeletons. you draw skeletons.

## Installation

### PIP

```bash
pip install easy-dwpose
```

### From source

```bash
git clone git@github.com:reallyigor/easy_dwpose.git
cd easy_dwpose
pip install -e .
```

## Quickstart

### In you own .py scrip or in Jupyter

```python
import torch
from PIL import Image

from easy_dwpose import DWposeDetector

# You can use different devices: "cpu", "cuda:0", or "mps" (Apple Silicon)
device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
detector = DWposeDetector(device=device)
input_image = Image.open("assets/pose.png").convert("RGB")

# Generate skeleton image
skeleton = detector(input_image, output_type="pil", include_hands=True, include_face=True)
skeleton.save("skeleton.png")

# Or get pose data as JSON
pose_json = detector(input_image, output_type="json", draw_pose=None)
print(pose_json)

# Or get pose data as Python dictionary
pose_dict = detector(input_image, output_type="dict", draw_pose=None)
print(pose_dict.keys())  # ['bodies', 'body_scores', 'hands', 'hands_scores', 'faces', 'faces_scores']
```

<table align="center">
    <tr>
      <th align="center">Input</th>
      <th align="center">Output</th>
    </tr>
    <tr>
        <td align="center">
          <br />
          <img src="./assets/pose.png"/>
        </td>
        <td align="center">
          <br/>
          <img src="./assets/skeleton.png"/>
        </td>
    </tr>
</table>

### On a video

```bash
python scripts/inference_on_video.py --input assets/dance.mp4 --output_path result.mp4

# Or generate JSON pose data for each frame
python scripts/inference_on_video.py --input assets/dance.mp4 --output_path result.mp4 --output_type json

# Or generate both video and JSON
python scripts/inference_on_video.py --input assets/dance.mp4 --output_path result.mp4 --output_type both
```

<table align="center">
    <tr>
      <th align="center">Input</th>
      <th align="center">Output</th>
    </tr>
    <tr>
        <td align="center">
          <br />
          <img src="./assets/dance.gif"/>
        </td>
        <td align="center">
          <br/>
          <img src="./assets/skeleton.gif"/>
        </td>
    </tr>
</table>

### On a folder of images

```bash
python scripts/inference_on_folder.py --input assets/ --output_path results/

# Or generate JSON pose data for each image
python scripts/inference_on_folder.py --input assets/ --output_path results/ --output_type json

# Or generate both images and JSON
python scripts/inference_on_folder.py --input assets/ --output_path results/ --output_type both
```

## Output Types

Easy DWPose supports multiple output formats:

### Image Outputs
- **`"pil"`**: Returns a PIL Image object (default for drawing functions)
- **`"np"`**: Returns a NumPy array (RGB format)

### Data Outputs
- **`"dict"`**: Returns pose data as a Python dictionary
- **`"json"`**: Returns pose data as a JSON string

### Pose Data Structure

When using `output_type="dict"` or `output_type="json"`, the returned data contains:

```python
{
    "bodies": [...],        # Body keypoints (18 points per person)
    "body_scores": [...],   # Confidence scores for body keypoints
    "hands": [...],         # Hand keypoints (21 points per hand, left then right)
    "hands_scores": [...],  # Confidence scores for hand keypoints
    "faces": [...],         # Face landmarks (68 points per face)
    "faces_scores": [...]   # Confidence scores for face landmarks
}
```

**Note**: To get pose data without generating images, set `draw_pose=None` in your call.

### Enhanced Part-Specific Detection

Easy DWPose now supports extracting specific body parts independently:

```python
from PIL import Image
from easy_dwpose import DWposeDetector

detector = DWposeDetector()
input_image = Image.open("assets/pose.png").convert("RGB")

# Get only body keypoints (no hands or face)
body_only = detector.get_body_only(input_image, output_type="dict")
print(body_only.keys())  # ['bodies', 'body_scores']

# Get only hand keypoints
hands_only = detector.get_hands_only(input_image, output_type="json")

# Get only face landmarks
face_only = detector.get_face_only(input_image, output_type="dict")

# Get whole body (all parts) - equivalent to default behavior
wholebody = detector.get_wholebody(input_image, output_type="pil")

# Custom combinations using the main detector
body_and_hands = detector(
    input_image,
    output_type="dict",
    include_body=True,
    include_hands=True,
    include_face=False
)

face_and_hands = detector(
    input_image,
    output_type="json",
    include_body=False,
    include_hands=True,
    include_face=True
)
```

#### Available Methods:
- **`get_body_only()`**: Extract only body pose (18 keypoints)
- **`get_hands_only()`**: Extract only hand keypoints (21 points per hand)
- **`get_face_only()`**: Extract only face landmarks (68 points)
- **`get_wholebody()`**: Extract all parts (body + hands + face)

#### Flexible Output Control:
You can also use the main detector with granular control:

```python
# Mix and match any combination of parts
result = detector(
    input_image,
    output_type="dict",  # or "json", "pil", "np"
    include_body=True,   # Include body keypoints
    include_hands=False, # Skip hand keypoints
    include_face=True    # Include face landmarks
)
```

This gives you complete flexibility to extract exactly the pose data you need, reducing output size and processing time when you only need specific body parts.

### Custom skeleton drawing

By default, we use standart skeleton drawing function but several projects change it (e.g. [MusePose](https://github.com/TMElyralab/MusePose)). Modify it or write your own from scratch!

```python
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose

detector = DWposeDetector(device="cpu")
input_image = Image.open("assets/pose.png").convert("RGB")

skeleton = detector(input_image, output_type="pil", draw_pose=draw_pose_musepose, draw_face=False)
skeleton.save("skeleton.png")
```

## Device Support

Easy DWPose supports multiple compute devices:

- **CPU**: `device="cpu"` - Works on all systems
- **CUDA**: `device="cuda"` or `device="cuda:0"` - For NVIDIA GPUs
- **MPS**: `device="mps"` - For Apple Silicon devices (M1/M2/M3 chips)

### Apple Silicon (MPS) Support

On Apple Silicon devices, you can use the MPS (Metal Performance Shaders) backend for improved performance:

```python
import torch
from easy_dwpose import DWposeDetector

# Check if MPS is available
if torch.backends.mps.is_available():
    detector = DWposeDetector(device="mps")
    print("Using MPS device for optimized performance on Apple Silicon!")
else:
    detector = DWposeDetector(device="cpu")
    print("MPS not available, using CPU")
```

**Note**: Easy DWPose utilizes CoreML ExecutionProvider for ONNX Runtime on Apple Silicon devices, providing native hardware acceleration through the Neural Engine, GPU, and CPU. This offers significant performance improvements compared to CPU-only execution.

## Acknowledgement

We thank the original authors of the [DWPose](https://github.com/IDEA-Research/DWPose) for their incredible models!

Thanks for open-sourcing!
