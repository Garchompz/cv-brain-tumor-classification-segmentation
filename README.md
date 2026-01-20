# BrainSight: Efficient Brain Tumor Segmentation Utilizing Computer Vision
This final project is part of the Computer Vision (COMP7116001) streaming course during the 2025/2026 odd semester period in BINUS University.

Created by Group 17 Computer Vision class LE01:
1. Kent Amadeo Timotheus - 2702227025
2. Theodore Zachary - 2702244100
3. Albertus Edbert Chandrajaya - 2702345440

This project covers brain tumor segmentation utilizing classical machine learning, random forest, to segment parts of an MRI scan which is indicated as a tumor. The output of the model is a binary mask of the tumor area (white) and non tumor area (black). This repository also covers the report, dataset, app code, and experiment notebook to generate the model.
## Experiment results
The model achieved the following metrics on evaluation: 
- Accuracy: 0.9514
- Dice/F1 Score: 0.9527
## How to run the application
### 1. Extract the model from the zip files
The model is compressed into two separate zip files, if extracted correctly, it should have create a single joblib file. Here's an example of how to extract it (not associated with this repository): [Extracting Multi-Part Zip Files](https://knowledge.navvis.com/docs/how-to-extract-multi-part-zip-files-with-z01-extension)
### 2. Run backend (Python)
Start the flask app, after ensuring that the model is in the right directory according to the code.
```python
# run backend (adjust path if working dir is backend)
python backend/app.py
```
### 3. Run frontend (Vite + React)
Install Node dependencies and start the dev server:
```cmd
cd app\frontend\main-app
npm install
npm run dev
```
