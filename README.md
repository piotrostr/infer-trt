# yolo-tensorrt
Interface for TensorRT engines inference along with an example of YOLOv4 engine being used.

<br />
<div align="center">
  <img width="500" src="https://user-images.githubusercontent.com/63755291/165267626-1d2ddeb7-bf57-4640-ac33-7f8b3a9bf72d.png">
</div>

### System Requirements

- amd64/linux architecture
- Nvidia GPU with Tensor cores

### Usage

Installation is quite complex so it is adviced to use the pre-build nvcr.io TensorRT container.

```bash
docker pull nvcr.io/nvidia/tensorrt:22.03-py3
```

The example uses the opencv library, which can be built using the official instructions which can be found in the <a href="https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html">official docs</a>. 
Building it even with 8-core-cpu takes quite long so I'd advise to install the binary from conda-forge. 
```
conda install -y -c conda-forge/label/gcc7 opencv
```
In order to obtain the TensorRT engine for a given model the ```trtexec``` tool can be used to make an export from onnx weights file.

```bash
 ./trtexec \
     --onnx=./yolo.onnx \
     --best \
     --workspace=1024 \
     --saveEngine=./yolo.trt \
     --optShapes=input:1x3x416x416 
```

The ```trt_model.py``` contains a base class to be used for inheritance. Once the preprocesing and postprocessing methods are overriden to match the steps required per given model, it is ready for inference.
with its high-level api:

```python
import cv2
import tensorrt as trt
import pycuda.autoinit

from yolo import YOLO

yolo = YOLO(trt.Logger())
img = cv2.imread('some_img.png')
labels, confidences, bboxes = yolo(img)
```
