# Model: YOLOv9 with DenseNet121 backbones

This directory contains the modified YOLOv9 implementation for mammographic lesion detection. The original YOLOv9-c GELAN backbone is replaced with DenseNet121; the YOLOv9 detection neck and head are retained. Two initialization strategies are compared with the same detection setup:

|Model configuration|Backbone initialization|Purpose|
|-|-|-|
|`yolov9-rad-densenet121.yaml`|RadImageNet|Main experiment|
|`yolov9-imagenet-densenet121.yaml`|ImageNet|Controlled pretraining comparison|

The detector predicts two lesion classes: `0` = benign and `1` = malignant. Input mammograms are preprocessed into three-channel, letterboxed 1024 × 1024 images. The experiments used AdamW and froze the backbone for the first 15 epochs before fine-tuning in the modified training script.

## Directory guide

* `train\_freeze\_15.py`: modified YOLOv9 training entry point with the 15-epoch backbone freeze/unfreeze schedule.
* `models/common.py`: DenseNet121 backbone modules and pretrained-weight loading.
* `models/yolo.py`: model parser support for the custom backbone.
* `models/detect/yolov9-rad-densenet121.yaml` and `models/detect/yolov9-imagenet-densenet121.yaml`: architecture configurations.
* `data/hyps/`: training configurations, including `hyp.overfit-test.yaml`, `hyp.transfer-low-lr.yaml`, `hyp.tune-B-noaug.yaml`, `hyp.tune-C-clsweight.yaml`, and `hyp.tune-D-combined.yaml`.

## Prerequisites and data

From the repository root, install the dependencies in `requirements.txt`. The experiments were run in Google Colab with an NVIDIA L4 GPU. The committed `model/` folder contains the files changed for this project. Before running training or evaluation, clone [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) separately and add its missing modules, including `model/train.py`, `model/val.py`, and `model/utils/`, to your local working copy. Keep the project versions of `model/models/common.py` and `model/models/yolo.py`; do not overwrite them with upstream files. Also preserve the upstream license and use a compatible revision. Upstream files added locally should not be described as committed project files in the repository tree.

CBIS-DDSM, INbreast, and model weights are to be kept outside version control. The CBIS-DDSM dataset is described by Lee, R.S. et al. (2017), “A curated mammography data set for use in computer-aided detection and diagnosis research,” *Scientific Data*, 4, 170177. The INbreast data used for external validation came from [INbreast 2012 on Kaggle](https://www.kaggle.com/datasets/tommyngx/inbreast2012). The notebook expects the processed datasets in sibling folders of the repository by default:

```text
parent-of-repository/
├── final\_project\_mammography/
│   ├── model/
│   └── notebooks/
├── YOLO\_labelled\_2c/
│   ├── dataset.yaml
│   ├── dataset\_final.yaml
│   ├── train/
│   ├── val/
│   └── test/
└── inbreast\_YOLO\_labelled\_2c/
    └── dataset.yaml
```

You may choose another location by setting `MAMMO\_DATA\_ROOT` to the parent of the two processed-data folders and `MAMMO\_REPO\_ROOT` to the cloned repository. Edit each local dataset YAML so its `train`, `val`, and `test` entries resolve to actual images and labels. Preserve the original CBIS-DDSM test set as the held-out test set. `dataset\_final.yaml` was used for final training on combined train and validation data; check its `train` entry and confirm its `test` entry still points to the held-out set. Dataset preparation is documented in `../preprocessing/`.

### Pretrained weights

The RadImageNet model starts from the DenseNet121 weights distributed by the [RadImageNet project](https://github.com/BMEII-AI/RadImageNet). Obtain the source weights separately. The training and evaluation notebook contains the Keras-to-PyTorch conversion used in this project and expects the input file at `model/radimagenet\_weights/RadImageNet-DenseNet121\_notop.h5`; it writes `radimagenet\_densenet121\_selfconverted.pt` beside it. Check that the path referenced by your backbone loader in `models/common.py` matches the converted checkpoint. The ImageNet backbone uses torchvision's pretrained DenseNet121 weights and may download them on first use. Do not commit either set of weights or the trained `.pt` checkpoints.

## Training workflow

Run commands from the repository root. These examples reflect the training and evaluation notebook; replace `/path/to/YOLO\_labelled\_2c` with the location of your processed data.

|Experiment stage|Training entry point|
|-|-|
|Developing a model better than the baseline (initial 15-epoch frozen-backbone run)|Upstream `model/train.py` with `--freeze 1`|
|Overfitting study, hyperparameter variants A–D, and final RadImageNet and ImageNet runs|Modified `model/train\_freeze\_15.py`, which freezes the backbone for the first 15 epochs before fine-tuning|

Add upstream `train.py` locally before reproducing the initial experiment. Only `train\_freeze\_15.py` is listed in the committed project tree because it contains the project-specific training change.

### Tune the RadImageNet model

Variant A used `hyp.transfer-low-lr.yaml`. Variants B, C, and D changed augmentation, classification loss weighting, or both. Variant C was selected from validation performance for final training.

```bash
python model/train\_freeze\_15.py \\
  --img 1024 --batch 4 --epochs 100 \\
  --data /path/to/YOLO\_labelled\_2c/dataset.yaml \\
  --cfg model/models/detect/yolov9-rad-densenet121.yaml \\
  --hyp model/data/hyps/hyp.tune-C-clsweight.yaml \\
  --weights '' --optimizer AdamW --name tune\_C\_clsweight
```

For the other tuning runs, use `hyp.transfer-low-lr.yaml` with `--name tune\_A\_baseline`, `hyp.tune-B-noaug.yaml` with `--name tune\_B\_noaug`, or `hyp.tune-D-combined.yaml` with `--name tune\_D\_combined`. The notebook also contains the preliminary better-than-baseline run (`train.py`) and the overfitting study (`train\_freeze\_15.py`) using `hyp.overfit-test.yaml`; those are historical steps, not required to evaluate the final checkpoints.

### Train the selected configuration

The recorded final run used 68 epochs, chosen from the tuning run's combined validation loss, and trained on the combined training and validation images:

```bash
python model/train\_freeze\_15.py \\
  --img 1024 --batch 4 --epochs 68 \\
  --data /path/to/YOLO\_labelled\_2c/dataset\_final.yaml \\
  --cfg model/models/detect/yolov9-rad-densenet121.yaml \\
  --hyp model/data/hyps/hyp.tune-C-clsweight.yaml \\
  --weights '' --optimizer AdamW --name final\_radimagenet\_densenet121
```

For the comparison run, use the same command with `--cfg model/models/detect/yolov9-imagenet-densenet121.yaml` and `--name imagenet\_densenet121`. Training outputs are expected under `model/runs/train/<run-name>/`. The notebook loads `last.pt` for both final models, so retain those locally if reproducing its results. A resumed experiment may have a numbered run directory; point the notebook at the actual checkpoint path if yours differs.

## Evaluation

The complete evaluation, threshold selection, DeLong test, INbreast validation, and error analysis are in [`../notebooks/mammography\_training\_and\_evaluation.ipynb`](../notebooks/mammography_training_and_evaluation.ipynb). After adding the upstream `val.py` and `utils/` locally, calculate detection metrics on the held-out CBIS-DDSM test split:

```bash
python model/val.py \\
  --data /path/to/YOLO\_labelled\_2c/dataset\_final.yaml \\
  --weights model/runs/train/final\_radimagenet\_densenet121/weights/last.pt \\
  --img 1024 --batch 4 --task test --name test\_eval\_radimagenet\_densenet121 --verbose
```

Repeat with the ImageNet checkpoint to compare mAP, precision, and recall. The notebook derives an image-level malignant score from the highest malignant detection confidence. Accuracy and F1 use the fixed **0.046104** threshold selected from the RadImageNet tuning variant's validation predictions; the held-out test set is not used to choose the threshold.

|Held-out CBIS-DDSM test model|Image-level AUC-ROC|mAP@0.5|mAP@0.5:0.95|
|-|-:|-:|-:|
|RadImageNet DenseNet121|0.6288|0.199|0.0913|
|ImageNet DenseNet121|0.6463|0.208|0.0891|

The paired DeLong comparison yielded *p* = 0.3027; this experiment did not establish an AUC advantage for RadImageNet pretraining. On INbreast, the final RadImageNet model achieved image-level AUC-ROC 0.7549 but lesion mAP@0.5 of 0.00792. See the notebook for the remaining metrics and interpretation of the external validation result. These models and the demonstration interface are for research use only, not clinical decision-making.

## Source and attribution

This project adapts [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) and uses pretraining from [BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet). Preserve the upstream license and attribution files when distributing modified upstream source. Consult the root README for the dataset and paper references.

