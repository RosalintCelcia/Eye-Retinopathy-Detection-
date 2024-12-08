# Eye-Retinopathy-Detection-
This repository contains the implementation of a deep learning model to detect eye retinopathy using segmentation techniques. The project leverages the UNet architecture with an EfficientNetB4 backbone to perform optic disc segmentation, enabling precise detection for medical imaging purposes.

# Project Overview
Eye retinopathy is a condition that can lead to vision impairment or loss if untreated. Early detection is crucial for timely intervention. This project uses deep learning to segment and analyze retinal images, focusing on the optic disc region, which is essential for detecting abnormalities.

# Features
Data Preprocessing:
  - Resizes and normalizes retinal images and ground truth masks to a uniform size of 512x512.
  - Encodes ground truth masks for binary classification.
Model Architecture:
  - Utilizes UNet with an EfficientNetB4 encoder pre-trained on ImageNet.
  - Supports segmentation with sigmoid activation for binary classification.
Metrics and Evaluation:
 - Monitors training and validation performance using metrics like:
      Binary Accuracy
      Precision
      Recall
      Intersection over Union (IoU)
      F1-Score
      Area Under Curve (AUC)
   
# Visualization:
  - Provides comparative plots for input images, ground truth masks, and predicted masks.
  - Displays training and validation loss/accuracy curves.

# Dependencies
The project is implemented in Python and requires the following libraries:
  TensorFlow/Keras
  Segmentation Models Library (segmentation-models)
  NumPy
  OpenCV
  Matplotlib
  Google Colab (optional, for cloud-based execution)

# Usage
1.Dataset Preparation:
  Download and place the retinal images and corresponding ground truth masks into their respective directories.
  Ensure the directory structure matches the code paths.
2.Install Dependencies: Install the required libraries using pip:

pip install tensorflow keras segmentation-models matplotlib opencv-python
3.Run the Code: Execute the script to preprocess the data, train the model, and evaluate its performance:
python eye_retinopathy_detection.py

# Visualizations:
  The script generates and saves:
    Training and validation loss/accuracy plots.
    Comparative visualizations of input images, ground truth masks, and predicted masks.

# Results
  Performance Metrics:
    - The model achieves high precision, recall, and IoU scores, indicating its effectiveness in segmenting optic disc regions.

  Visualizations:
    - Detailed plots and comparative masks validate the model's accuracy.

# Future Scope
- Integrate multi-class segmentation for other retinal conditions.
- Use attention-based mechanisms to improve segmentation accuracy.
- Expand to other medical imaging datasets.
  
Feel free to fork and contribute to this repository. Feedback and suggestions are welcome!
