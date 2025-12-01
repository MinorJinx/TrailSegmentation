# TrailSegmentation

TrailSegmentation detects and maps walking/hiking/biking trails in satellite imagery using deep learning-based pixel-wise segmentation, with tooling to train, evaluate, and benchmark multiple architectures and backbone encoders.

------------------------------------------------------------
### 1) Thesis overview
------------------------------------------------------------
This master’s thesis explores an under-studied segmentation problem: identifying narrow, partially occluded recreational trails (walking, hiking, biking) from satellite images. Compared to roads, trails are thinner, have weaker boundaries, and are often hidden by foliage/shadows making them easy to confuse with dirt roads and other linear features.

The thesis work includes:
- Building a curated trail dataset from satellite imagery and creating ground-truth masks
- Training multiple state-of-the-art CNN segmentation models
- Evaluating models using segmentation-focused metrics (e.g., IoU and BDE) plus practical constraints:
  model size, model load time, and inference time
- Real-world testing on a familiar area and documenting failure modes (occlusions, dirt-road confusion)
- Future directions such as post-processing to connect fragmented predictions and multi-class labeling

------------------------------------------------------------
### 2) What the code does
------------------------------------------------------------
This repo contains an experiment/benchmark framework (TensorFlow/Keras) to:
- Load trail images + binary masks and create train/val/test datasets
- Apply normalization and optional Albumentations data augmentation
- Build and train multiple segmentation architectures
- Swap backbone encoders (ResNet, MobileNet, EfficientNetV2 variants)
- Track training metrics and save models + training history
- Measure model load time, per-image inference time, and test IoU, and export results for comparison

------------------------------------------------------------
### 3) Models and backbones
------------------------------------------------------------
Models compared include:
- U-Net, U-Net++, U-Net 3+
- SegNet
- LinkNet
- DeepLabV3
- Mask R-CNN (plus a custom variant)

Backbones include common Keras application encoders, such as:
- ResNet50 / ResNet101
- MobileNet / MobileNetV2
- EfficientNetV2 variants

------------------------------------------------------------
### 4) Repo contents
------------------------------------------------------------
Files in this repo:
- model_manager.py
  Core training/evaluation/benchmarking logic and model implementations.
- Main.ipynb
  Example workflow for selecting models/backbones, running experiments, and plotting results.

Expected local folders (not included unless you add them):
- segmentation/
  - images/ and masks/             (train/val)
  - test/images/ and test/masks/   (hold-out test set)
- models/
  Saved model files and training history.

------------------------------------------------------------
### 5) Basic usage
------------------------------------------------------------
1) Install dependencies from requirements.txt

2) Add data:
Place your images and binary masks into the expected folder structure.
Images and masks should align by filename.

3) Train:
python model_manager.py --model unet_3plus --backbone efficientnetv2s --epochs 50 --batch_size 4

4) Benchmark (load time, inference time, test IoU):
python model_manager.py --model unet_3plus --backbone efficientnetv2s --testing_mode --output_file model_performance.json

5) Notebook workflow:
Open Main.ipynb to run experiments interactively and generate plots/comparison charts.

------------------------------------------------------------
### 6) Data note
------------------------------------------------------------
This repo does not include the satellite imagery dataset. If you recreate the dataset yourself, make sure you comply with your imagery provider’s terms. For fully redistributable workflows, consider using open satellite sources (e.g., Sentinel/Landsat/USGS) and regenerating masks accordingly.

------------------------------------------------------------
### 7) Citation
------------------------------------------------------------
If you use this repo for academic work, please cite the associated master’s thesis:
[https://www.proquest.com/openview/032ef75a1e34ff48487ca757625cde89](https://www.proquest.com/openview/032ef75a1e34ff48487ca757625cde89)
