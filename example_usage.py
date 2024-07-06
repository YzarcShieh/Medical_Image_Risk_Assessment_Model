import pandas as pd
from sklearn.model_selection import train_test_split
from main import main

# Read data
df = pd.read_csv('./example_data/image_labels_info.csv')
y_col_name='label'

# Split data into trainvalid & test
trainval, test_data = train_test_split(
    df, test_size = 0.2, stratify = df[y_col_name], random_state=1)

# Further split trainvalid into train & valid
train_data, valid_data = train_test_split(
    trainval, test_size = 0.2, stratify = trainval[y_col_name], random_state=1)

# Further split traindata into train & xgb_train
train_data_densenet, train_data_xgb = train_test_split(
    train_data, test_size=0.5, stratify = train_data[y_col_name], random_state=1)

# Define model parameters for DenseNet and XGBoost
model_params_densenet = {
    'max_epochs': 5,
    'batch_size': 64,
    'learning_rate': 1e-5
}

model_params_xgb = {
    'gamma': 0,
    'max_depth': 6,
    'subsample': 0.7,
    'min_child_weight': 1,
    'scale_pos_weight': 1,
    'eta': 0.3
}

# Run the main function
main(train_data_densenet, train_data_xgb, valid_data, test_data, y_col_name = y_col_name, model_params_densenet=model_params_densenet, model_params_xgb=model_params_xgb)
