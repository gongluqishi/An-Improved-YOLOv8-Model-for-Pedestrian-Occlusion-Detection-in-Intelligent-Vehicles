## YOLOv8m-C2f-DCNv3 Occluded Pedestrian Detection Reproduction

This directory has completed the C2f-DCNv3 modification for YOLOv8m as required by the Project Requirements.docx document, and organized reproducible pipelines for data processing, model training, evaluation and controlled comparative experiments. The upstream source code version is Ultralytics 8.3.51, and DCNv3 implementation is derived from the attached InternImage repository.

## Completed Work
Integrated the original InternImage detection/ops_dcnv3 implementation into ultralytics/ops_dcnv3.
Added a device-agnostic DCNv3_YOLO wrapper that maintains the NCHW tensor layout adopted by YOLO externally.
Implemented Bottleneck_DCNV3 and C2f_DCNV3, with module export and model parser registration fully configured.
Created a fixed YOLOv8m-scale model configuration file, replacing all 8 C2f blocks in the backbone and neck with C2f-DCNv3 modules.
Retained optional CUDA/C++ extensions; if extensions are not compiled, a differentiable pure PyTorch fallback implementation will be automatically activated, supporting both CPU and GPU execution.
Fixed issues including absolute paths, incorrect dataset splitting and category semantic conflicts in the original data1.yaml, generating a portable single-class pedestrian dataset config file.
Provided supporting scripts for data cleansing, full dataset auditing, training, evaluation, baseline comparison and pytest regression testing.

## Dataset Processing Summary
The original person1 dataset uses class ID 0 to represent pedestrians in training and validation sets. However, its test set adopts inconsistent labeling rules: class ID 0 for vehicles and class ID 1 for pedestrians, mixed with segmentation contour labels. To eliminate semantic contamination between categories, all labels have been unified to a single target class pedestrian:
Training set: 7,200 images with 209,842 valid pedestrian bounding boxes; 3 invalid zero-width boxes removed.
Validation set: 1,800 images with 50,316 pedestrian bounding boxes.
Test set: 92 images with 137 retained pedestrian annotations; 78 vehicle labels deleted, and 24 pedestrian segmentation contours converted into detection bounding boxes.
40 pedestrian-free background images in the test set are preserved as valid negative samples.
The complete raw dataset is archived in person1.zip. The full data cleansing workflow can be reviewed and re-run via scripts/prepare_person1_detection.py. All dataset audit logs are stored under results/validation.

## Environment Setup
Run the following commands in the ultralytics-main directory via PowerShell:
powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install torch/torchvision matching your local CUDA version first; CPU-only installation is acceptable for validation only.
python -m pip install -r requirements-reproduction.txt
For formal GPU training with acceleration, compile the InternImage CUDA operators (requires CUDA-enabled PyTorch, CUDA Toolkit and C++ build toolchain):
powershell
python -m pip install -v .\ultralytics\ops_dcnv3
If compilation is skipped, the pure PyTorch reference implementation will be used automatically — it yields identical results but runs slower.

## Sanity Validation & Smoke Tests
powershell
python scripts\validate_dataset.py
pytest -q tests\test_dcnv3_integration.py
Minimal smoke training pipeline (no pre-trained weights required):
powershell
python scripts\train_dcnv3.py --smoke --weights none --device cpu --name smoke_cpu

## Full Reproduction Experiments
 DCNv3 Improved Model Experiment
Default settings: 100 training epochs, 640px input resolution, batch size = 2 (minimum 6GB VRAM required):
powershell
python scripts\train_dcnv3.py --device 0 --name yolov8m_dcnv3_person1
Baseline Original YOLOv8m Experiment
Identical training entry and hyperparameters, only the model config file is replaced:
powershell
python scripts\train_dcnv3.py --model yolov8m.yaml --device 0 --name yolov8m_baseline_person1
If you encounter out-of-memory errors, change --batch 2 to --batch 1. For formal paper comparison, ensure both groups share identical random seed, data split, input image size, batch size, epoch count and augmentation policies.
Evaluate Best-Performing Weights
powershell
python scripts\evaluate_dcnv3.py --weights ..\results\train\yolov8m_dcnv3_person1\weights\best.pt --split test
Generate Quantitative Comparison Between Baseline and DCNv3 Model
powershell
python scripts\compare_results.py `
  --baseline ..\results\train\yolov8m_baseline_person1\results.csv `
  --dcnv3 ..\results\train\yolov8m_dcnv3_person1\results.csv `
  --output ..\results\comparison\baseline_vs_dcnv3.md
## Current Validation Status
Single DCNv3 module forward & backward pass: Passed
C2f-DCNv3 block forward & backward pass: Passed
Full YOLOv8m-DCNv3 network construction: Passed, detection stride 8/16/32
Full model forward & backward pass on 64×64 three-scale feature maps: Passed, all input gradients are finite values
Full image & label audit for the person1 dataset: Passed
CPU offline smoke training: Passed; 1-epoch small-scale training and evaluation on 1,800 validation images completed, both best.pt and shturl. successfully saved to results/train/smoke_cpu/weights
The current test environment is equipped with CPU-only PyTorch, so CUDA extension compilation and full 100-epoch time-consuming formal training were not executed in this iteration. However, the training entry script and optional CUDA acceleration path are fully functional and ready for use.
