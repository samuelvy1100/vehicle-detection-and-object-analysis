# Computer Vision: Vehicle Detection and Object Analysis

## Program Overview
This project implements a lane-aware object detection pipeline using a pre-trained YOLOv8 model applied to real-world driving video footage from Miami, FL, to New York City, NY. The system extracts video frames, performs object detection at scale, overlays lane geometry, and computes normalized spatial metrics to reason about object proximity and position relative to the driving lane. The program demonstrates how modern computer vision models can be integrated with geometric reasoning to support traffic analysis, autonomous driving research, and intelligent transportation systems.

---

## Focus
The primary areas of interest are:

- Apply a pre-trained deep learning object detector (YOLOv8) to real traffic video.
- Detect and classify objects such as cars, trucks, buses, pedestrians, and traffic infrastructure.
- Introduce lane subdivision and center-line geometry to contextualize detections.
- Normalize bounding box coordinates to enable spatial reasoning across frames.
- Identify nearby objects based on relative bounding box area.

Rather than training a model from scratch, the emphasis is on deployment, interpretation, and post-processing of deep learning outputs in a realistic scenario.

---

## Methodology & Implementation

### Demo Video  
The following link contains the driving footage used as input for frame extraction and object detection in this project.

[(YouTube) Miami to NYC Driving Footage](https://www.youtube.com/watch?v=9qy4lExIetk)

**Preview GIF:**

<p align="center">
  <img src="media/video_preview.gif" width="700" alt="YOLOv8 lane-aware object detection preview">
</p>

### 1. Video Frame Extraction
- Video footage is sampled at fixed time intervals (every 10 seconds).
- Frames are extracted using OpenCV and saved for batch inference.
- This approach balances temporal coverage and computational efficiency.

### 2. Object Detection with YOLOv8
- A pre-trained `yolov8s` model from the Ultralytics framework is used.
- Each extracted frame is passed through the model to obtain:
  - Object class labels
  - Bounding boxes
  - Confidence scores
- Annotated frames are saved automatically for inspection.

### 3. Lane Geometry Overlay
- A virtual center lane line is defined using proportional image width.
- Additional diagonal boundaries approximate lane structure.
- Lane overlays are plotted alongside YOLO detections to contextualize traffic flow.

### 4. Spatial Normalization and Feature Engineering
For each detected object:
- Bounding box centers are normalized to `[0, 1]` image coordinates.
- Bounding box area is normalized relative to frame size.
- A proximity flag (`nearby`) is computed based on normalized area thresholds.

All detections are aggregated into structured tabular data for analysis.

### 5. Large-Scale Batch Inference
- The model is applied to hundreds of extracted frames.
- Detection summaries (object counts per frame) are logged.
- Results illustrate the variability of traffic density and object composition over time.

---

## Results
- The system reliably detects vehicles, pedestrians, and infrastructure across diverse frames.
- Larger bounding box areas consistently correspond to objects closer to the camera.
- Lane geometry improves interpretability by highlighting which objects occupy the ego lane.
- The pipeline demonstrates strong inference performance without fine-tuning, validating the robustness of modern pre-trained vision models.

---

## Applications
The methodologies and data extraction techniques used in this project are applicable into several industrial and research-based sectors including but not limited to:

- **Autonomous Driving**: Lane-aware object localization supports collision avoidance and path planning.
- **Traffic Monitoring**: Automated vehicle counting and classification can inform congestion analysis.
- **Smart Cities**: Scalable video analytics for infrastructure planning and safety monitoring.
- **Computer Vision Education**: Demonstrates end-to-end deployment of deep learning models with geometric reasoning.

---

## Technologies Used
- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy & Pandas
- Matplotlib
- Jupyter Notebook

---

## Notes
- This project focuses on inference and spatial reasoning rather than model training.
- Lane boundaries are geometrically approximated and not learned from data.
- The pipeline is modular and can be extended to real-time video streams.


