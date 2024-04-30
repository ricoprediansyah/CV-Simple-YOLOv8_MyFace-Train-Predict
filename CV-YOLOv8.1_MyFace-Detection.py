## 1. Install Requirements YOLOv8.1
%pip install ultralytics
import ultralytics
ultralytics.checks()


## 2. Import Labeling Roboflow
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="PZMfN1gExhcqfGkkkd43")
project = rf.workspace("rico-prediansyah").project("myface-v1gul")
version = project.version(1)
dataset = version.download("yolov8-obb")

## 3. set up environment
import os
os.environ["YOLO8_DATA"] = "/content/MyFace-1"

## 4. Training
!yolo train model=yolov8n.pt data=/content/MyFace-1/data.yaml epochs=100 imgsz=640
# imgsz: define input image size
# epochs: define the number of training epochs. (Note: often, 3000+ are common here!)
# data: Our dataset locaiton is saved in the dataset.location
# weights: specify a path to weights to start transfer learning from. Here we choose the generic COCO pretrained checkpoint.

## 5. Predict
!yolo predict model=/content/runs/detect/train8/weights/best.pt source='/content/poltekes_02.jpg
# weights : location result after training
# source : location file for testing