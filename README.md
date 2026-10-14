# Crack Segmentation

An image-segmentation project for detecting cracks in surface images. It uses a
U-Net with a ResNet-34 encoder and provides Streamlit interfaces for viewing
predicted crack masks and overlays.

## Project contents

- `app.py` — Streamlit interface for railway-track crack segmentation.
- `app2.py` — Streamlit interface for metal-surface crack segmentation.
- `dataset.py` — dataset loading, augmentation, model training, and checkpoint
  saving.
- `test.py` — runs prediction on `dataset/input/1.bmp` and saves
  `pred_mask.png`.
- `dataset/input/` — training and test images.
- `dataset/output/` — binary training masks. Each mask must have the same base
  filename as its image and a `.png` extension.
- `dataset/depth_map/` — additional dataset files; not used by the current
  training script.

## Setup

Use a Python environment and install PyTorch for your hardware using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).
Then, from the project directory, install the remaining dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The scripts expect a trained model checkpoint named `best_model.pth` in the
project root. Obtain or train this checkpoint before running an app; model
weights and dataset files are excluded from Git by `.gitignore`.

## Run an app

Start either Streamlit interface from the project directory:

```powershell
streamlit run app.py
```

Or:

```powershell
streamlit run app2.py
```

Upload a BMP, JPG, or PNG image and select **Predict** to display the predicted
mask and overlay.

## Train the model

Place input images in `dataset/input/` and corresponding PNG masks in
`dataset/output/`. Then run:

```powershell
python dataset.py
```

The script trains a U-Net for 50 epochs and writes the best validation-IoU
checkpoint to `best_model.pth`.

## Run the test prediction

With `best_model.pth` available and `dataset/input/1.bmp` present, run:

```powershell
python test.py
```

The prediction is written to `pred_mask.png`.
