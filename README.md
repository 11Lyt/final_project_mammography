# Deep Learning Breast Lesion Detection using YOLOv9 and RadImageNet

## Overview
The goal of this project is to investigate whether integrating a RadImageNet-pretrained DenseNet121 CNN backbone into the YOLOv9 object detection framework improves the accuracy of mammogram abnormality detection compared with conventional ImageNet-pretrained models. 

## Architecture
<img width="214" height="344" alt="image" src="https://github.com/user-attachments/assets/4d83417c-66f1-4244-b581-0f58553d706b" />

## Dataset
### CBIS-DDSM
Main dataset used for model development and testing

### INBreast
Used for external validation

## Preprocessing
Dataset preprocessing was performed on Jupyter Notebook

### Preprocessing Pipeline
DICOM
→ intensity correction
→ percentile clipping
→ normalisation
→ RGB conversion
→ letterbox resize
→ bounding-box conversion
→ YOLO labels

## Model Training
Models were trained on Google Colab notebook with L4 GPU (Python 3)

## Evaluation Metrics
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1
- Accuracy
- ROC-AUC
- DeLong test

## Gradio Demo
A simple Gradio interface was developed to demonstrate model inference on mammography DICOM images.

The interface allows users to:

- Upload a mammogram in DICOM format
- Preview the preprocessed mammogram
- Run lesion detection using the trained YOLOv9 model
- Display predicted lesion bounding boxes, class labels and confidence scores

The uploaded DICOM image is processed using the same preprocessing pipeline used during model development, including photometric correction, percentile clipping, normalisation, RGB conversion and letterbox resizing.

### Demo Interface
<img width="1043" height="655" alt="image" src="https://github.com/user-attachments/assets/d06ebbcf-0161-4ea9-9962-d6bccc8c6dfa" />
<img width="985" height="513" alt="image" src="https://github.com/user-attachments/assets/3aba94e7-c8d7-4329-a1b5-c2ae99791174" />
<img width="1043" height="546" alt="image" src="https://github.com/user-attachments/assets/6651f98c-8810-4b66-9960-64c08f394bb2" />

> This interface is intended for research and demonstration purposes only and is not designed for clinical use.

## Repository Structure

```text
mammography-lesion-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── preprocessing/
│   ├── CBIS-DDSM_Preprocessing.ipynb
│   └── INbreast_Preprocessing.ipynb
│
├── model/
│   ├── README.md
│   ├── hyps/
│   │   ├── hyp.overfit-test.yaml
│   │   ├── hyp.transfer-low-lr.yaml
│   │   ├── hyp.tune-B-noaug.yaml
│   │   ├── hyp.tune-C-clsweight.yaml
│   │   └── hyp.tune-D-combined.yaml
│   ├── train.py
│   ├── common.py
│   ├── yolo.py
│   ├── yolov9-imagenet-densenet121.yaml
│   └── yolov9-rad-densenet121.yaml
│
├── notebooks/
│   └── mammography_training_and_evaluation.ipynb
│
└── app/
    └── model_user_interface.ipynb
```

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/11Lyt/final_project_mammography.git
cd mammography-lesion-detection
```
### 2. Create and activate a virtual environment
```bash
python -m venv venv
```
On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### 1. Dataset Preprocessing
Preprocessing is performed using the notebooks in the `preprocessing/` directory.

For CBIS-DDSM:
```text
preprocessing/CBIS-DDSM_Preprocessing.ipynb
```

For INBreast:
```text
preprocessing/INbreast_Preprocessing.ipynb
```
### 2. Model Training
The modified YOLOv9 implementation is located in the `model/` directory.

The RadImageNet model configuration:
```text
model/yolov9-rad-densenet121.yaml
```

The ImageNet comparison model:
```text
model/yolov9-imagenet-densenet121.yaml
```
### 3. Evaluation
Model evaluation, ROC-AUC analysis, threshold selection, DeLong testing and error analysis are provided in:
```text
notebooks/mammography_training_and_evaluation.ipynb
```

### 4. Gradio Application
Run the demonstration interface using:
```text
model_user_interface.ipynb
```

## References
- Lee, R.S. et al. (2017). **A curated mammography data set for use in computer-aided detection and diagnosis research.** *Scientific Data*, 4, 170177.

- Mei, X. et al. (2022). **RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning.** *Radiology: Artificial Intelligence*, 4(5).

- Rodriguez-Ruiz, A. et al. (2019). **Stand-Alone Artificial Intelligence for Breast Cancer Detection in Mammography: Comparison With 101 Radiologists.** *Journal of the National Cancer Institute*, 111(9), 916–922.

- Shen, L. et al. (2019). **Deep Learning to Improve Breast Cancer Detection on Screening Mammography.** *Scientific Reports*, 9, 12495.

- Wang, C.-Y., Yeh, I.-H. and Liao, H.-Y.M. (2024). **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.** arXiv:2402.13616.

### Source Repositories
- YOLOv9: `WongKinYiu/yolov9` at https://github.com/WongKinYiu/yolov9.git
- RadImageNet: `BMEII-AI/RadImageNet` at https://github.com/BMEII-AI/RadImageNet.git
