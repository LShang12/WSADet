# <div style="text-align: center;">WSADet: A Wavelet Scale-Aware UAV Object Detector for Complex Conditions</div>

---

## <div align="center">Documentation</div>

See below for a quickstart install and usage examples, and see our [Docs](https://docs.ultralytics.com/) for full documentation on training, validation, prediction and deployment.

<details open>
<summary>Install</summary>

Pip install the Ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart/).

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics) [![Ultralytics Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

</details>

<details open>
<summary>Usage</summary>

### CLI

YOLO may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` can be used for a variety of tasks and modes and accepts additional arguments, e.g. `imgsz=640`. See the YOLO [CLI Docs](https://docs.ultralytics.com/usage/cli/) for examples.

### Python

YOLO may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
```

See YOLO [Python Docs](https://docs.ultralytics.com/usage/python/) for more examples.

</details>

## Dataset
You can use your own dataset.

<details open>
  <summary><b>File structure</b></summary>

```
Your dataset
├── ...
├── images
|   ├── train
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
|   ├── val
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
└── labels
    ├── train
    |   ├── 1.txt
    |   ├── 2.txt
    |   └── ...
    └── val
        ├── 100.txt
        ├── 101.txt
        └── ...
```

</details>

## Citation

## Acknowledgement
Part of the code is adapted from previous works: [YOLOv11](https://github.com/ultralytics/ultralytics). We thank all the authors for their contributions.