# BTXRD Bone Lesion Classification

Source code for ROI-guided three-class bone lesion classification on radiographs using image features and clinical metadata.

Dataset:
The BTXRD dataset should be placed under data/btxrd/ with the following structure:

data/btxrd/images/
data/btxrd/Annotations/
data/btxrd/dataset.xlsx

Install:
pip install -r requirements.txt

Run:
python train.py

Outputs are written under the outputs/ directory.
