# Medical Image Risk Assessment

This project aims to provide a comprehensive approach to medical image classification and risk assessment using a combination of DenseNet121 and XGBoost. The pipeline includes training, feature extraction, and evaluation steps to achieve optimal model performance.

## Input Data Format
Users should prepare their dataset in a structured format:
- Create a CSV file containing image paths and their corresponding labels.
- Include the following columns in the CSV:
  - `image_file_name`: Path to each image file.
  - `label`: Numerical label indicating the risk category associated with each image.

Ensure that images are preprocessed and stored in a directory.

## Model Parameters
### DenseNet121 Parameters:
- `max_epochs`: Number of training epochs for DenseNet121. Default is 5.
- `batch_size`: Batch size for training. Default is 64.
- `learning_rate`: Learning rate for the optimizer. Default is 1e-5.

### XGBoost Parameters:
- `gamma`: Gamma parameter for regularization. Default is 0.
- `max_depth`: Maximum depth of a tree. Default is 6.
- `subsample`: Subsample ratio of the training instance. Default is 0.7.
- `min_child_weight`: Minimum sum of instance weight needed in a child. Default is 1.
- `scale_pos_weight`: Control the balance of positive and negative weights. Default is 1.
- `eta`: Learning rate. Default is 0.3.

Users can adjust these parameters to optimize model performance based on their dataset characteristics.

## Usage

To use this repository for the Medical Image Risk Assessment Model, follow these steps:

1. Clone the repository using the following git command:
   
  git clone https://github.com/yjhuang1119/Medical_Image_Risk_Assessment_Model.git

2. Ensure Python is installed on your machine.

3. Install required Python libraries by running:
  ```
  pip install -r requirements.txt
  ```

4. **Run the Example Directly:**
  You can run an example directly in `example_usage.py`.
  Optionally, modify the parameters:
  - `model_params_densenet`: Dictionary containing DenseNet121 model parameters (optional).
  - `model_params_xgb`: Dictionary containing XGBoost model parameters (optional).

5. **Use Your Own Data:**
  Prepare your dataset:
  Place images in a folder and create a CSV with image paths and labels.

  Update the CSV file in `example_usage.py` or use `main.py` directly if your data is split into training, validation, and test sets.

6. **Output Results:**
  The model generates the following outputs:

  Training Results:
  - `TrainResult.csv`: CSV file containing training loss and validation AUC for each epoch.
  - `TrainResult.tiff`: Plot of training loss and validation AUC.

  Feature Maps:
  Extracted feature maps for training, validation, and test data in CSV format.

  Evaluation Results:
  - `evaluation_results.csv`: CSV file containing AUC, accuracy, sensitivity, specificity, and F1 score.
  - `confusion_matrix.png`: Plot of the confusion matrix.

## File Summary
  - `train_densenet121.py`: Trains the DenseNet121 model on the provided training data and saves the best model based on validation AUC.
  - `extract_densenet121_featuremap.py`: Extracts feature maps from the DenseNet121 model for further processing with XGBoost.
  - `train_xgboost.py`: Trains an XGBoost model on the extracted DenseNet121 feature maps and validates it.
  - `evaluate_model.py`: Evaluates the trained XGBoost model using various performance metrics and plots the confusion matrix.
  - `main.py`: Main function to orchestrate the training, feature extraction, and evaluation steps.
  - `example_usage.py`: Example script demonstrating how to use the provided functions with a dataset.
  - `README.md`: This file provides project information and instructions.

