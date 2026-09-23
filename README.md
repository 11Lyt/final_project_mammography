# Deep Learning Breast Lesion Detection using YOLOv9 and RadImageNet

## Overview
The goal of this project is to investigate whether integrating a RadImageNet-pretrained DenseNet121 CNN backbone into the YOLOv9 object detection framework improves the accuracy of mammogram abnormality detection compared with conventional ImageNet-pretrained models and CNN models used in previous literature. 

## Architecture
<img width="214" height="344" alt="image" src="https://github.com/user-attachments/assets/4d83417c-66f1-4244-b581-0f58553d706b" />

## Dataset
### CBIS-DDSM
Main dataset used for model development and testing

### INBreast
Used for external validation

## Preprocessing Pipeline
DICOM
→ intensity correction
→ percentile clipping
→ normalisation
→ RGB conversion
→ letterbox resize
→ bounding-box conversion
→ YOLO labels

## Model Training
### Google Colab
[Open notebook]

## Evaluation Metrics
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1
- Accuracy
- ROC-AUC
- DeLong test

## Results
### CBIS-DDSM Test Set

### INBreast External Validation

## Gradio Demo

## Repository Structure

## Installation

## Usage

### Preprocessing
### Training
### Evaluation
### Running the Gradio interface
## Limitations

## References
