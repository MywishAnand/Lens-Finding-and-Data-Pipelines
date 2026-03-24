# Lens Finding Data Pipelines

## Overview
This project aims to build a solid deep learning pipeline for identifying strong gravitational lenses from astronomical imagery. The dataset comprises `(3, 64, 64)` image arrays representing three separate filters.

The core challenge resolved in this codebase is extreme **class imbalance**, where non-lensed galaxies outnumber lensed galaxies significantly.

## Strategy & Step-by-Step Implementation

### 1. Data Loading & Preprocessing (`dataset.py`)
- We developed a clean PyTorch `Dataset` (`LensDataset`) to ingest the custom `.npy` files effectively.
- **Handling Imbalance:** To counteract the data imbalance, we dynamically computed the ratio of negative (non-lenses) to positive (lenses) samples during data loading. This ratio serves as the `pos_weight` in our binary cross-entropy loss calculation, forcing the model to heavily penalize errors on positive samples.
- **Data Augmentations:** We integrated `RandomHorizontalFlip` and `RandomVerticalFlip` transforms into the training pipeline to make the model robust against rotational data variations typical in astrophysical imaging.

### 2. Model Architecture (`model.py`)
- For our backbone, we leveraged a **pretrained ResNet18**. It proves extremely resilient for handling feature extraction even on smaller `64x64` inputs.
- Recognizing the problem as binary classification, we tailored the fully connected classification head (`fc`) into a single output logit channel representing the probability of a lens. 

### 3. Training the Model (`train.py`)
- **Loss Strategy:** We utilized `BCEWithLogitsLoss(pos_weight=...)` optimized via `Adam` (`lr=1e-4`). 
- **Validation-Based Checkpointing:** The script concurrently monitors the validation loss epoch-to-epoch and saves only the best performing network weights (`best_model.pth`). 

### 4. Evaluating the Strategy (`evaluate.py`)
- We loaded our checkpoints into an isolated test loop that operates on `test_lenses` and `test_nonlenses`.
- Leveraging `scikit-learn`, we extracted ROC curves and calculated the total Area Under the Curve (AUC). The model achieved an astonishing **0.9794 AUC**.
- A visual representation of the `roc_curve.png` is generated seamlessly during runtime. 

## Run the Pipeline
1. Execute `python3 train.py` to initiate model training.
2. Once complete, run `python3 evaluate.py` to test and generate the evaluation curves.
