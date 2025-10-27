import pandas as pd
import numpy as np
from evaluate_model import prob_threshold, score
from train_densenet121 import densenet_model_train
from extract_densenet121_featuremap import extract_feature_maps
from train_xgboost import xgb_model_train
from train_tabpfn import tabpfn_model_train, plot_and_save_confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os

def main(densenet_train_data, xgb_train_data, valid_data, test_data, y_col_name='label', model_params_densenet=None, model_params_xgb=None, densenet_model_name = 'densenet_model.pth', xgb_model_name = 'xgb_model.pth'):
    """
    Main function to train, evaluate, and save results of the risk assessment model.

    Parameters:
    densenet_train_data (dict): Dictionary containing training data for DenseNet.
    xgb_train_data (dict): Dictionary containing training data for XGBoost.
    valid_data (dict): Dictionary containing validation data.
    test_data (dict): Dictionary containing test data.
    y_col_name (str, optional): Name of the target column. Defaults to 'PHENO'.
    model_params_densenet (dict, optional): Dictionary containing DenseNet model parameters. Defaults to None.
    model_params_xgb (dict, optional): Dictionary containing XGBoost model parameters. Defaults to None.

    Returns:
    None
    """
    # Default parameters for DenseNet121
    default_params_densenet = {
        'max_epochs': 10,
        'batch_size': 64,
        'learning_rate': 1e-5
    }

    if model_params_densenet is None:
        model_params_densenet = default_params_densenet

    # Default parameters for XGBoost
    default_params_xgb = {
        'gamma': 0,
        'max_depth': 6,
        'subsample': 0.7,
        'min_child_weight': 1,
        'scale_pos_weight': 1,
        'eta': 0.3
    }

    if model_params_xgb is None:
        model_params_xgb = default_params_xgb

    # Ensure output directory exists
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # DenseNet model training
    densenet_model_pth = os.path.join(output_dir, densenet_model_name)

    # DenseNet model training
    model, best_metric, best_metric_epoch = densenet_model_train(
        densenet_train_data.drop([y_col_name],axis=1).squeeze().tolist(), 
        densenet_train_data[y_col_name].to_list(), 
        valid_data.drop([y_col_name],axis=1).squeeze().tolist(), 
        valid_data[y_col_name].to_list(), 
        densenet_model_pth, 
        **model_params_densenet
    )

    # DenseNet feature extraction
    xgb_train_featuremap = extract_feature_maps(densenet_model_pth, xgb_train_data.drop([y_col_name],axis=1).squeeze().tolist(), xgb_train_data[y_col_name].to_list())
    valid_featuremap = extract_feature_maps(densenet_model_pth, valid_data.drop([y_col_name],axis=1).squeeze().tolist(), valid_data[y_col_name].to_list())
    test_featuremap = extract_feature_maps(densenet_model_pth, test_data.drop([y_col_name],axis=1).squeeze().tolist(), test_data[y_col_name].to_list())
    # Save feature maps to CSV
    xgb_train_featuremap.to_csv("./output/xgb_train_featuremap.csv", index=False)
    valid_featuremap.to_csv("./output/valid_featuremap.csv", index=False)
    test_featuremap.to_csv("./output/test_featuremap.csv", index=False)

    # XGBoost model training
    xgb_model_pth = os.path.join(output_dir, xgb_model_name)
    xgb_model = xgb_model_train(xgb_train_featuremap, valid_featuremap, xgb_model_pth, y_col_name=y_col_name, **model_params_xgb)

    # Model evaluation
    threshold_final = prob_threshold(xgb_model, valid_featuremap, y_col_name=y_col_name)
    evaluation_results = score(xgb_model, test_featuremap, threshold_final, y_col_name=y_col_name)
    evaluation_results.to_csv("./output/evaluation_results.csv", index=False)

    # ===== TabPFN: run AFTER XGBoost ===== 
    _train_fp = "./output/xgb_train_featuremap.csv"
    _valid_fp = "./output/valid_featuremap.csv"
    _test_fp  = "./output/test_featuremap.csv"

    _tabpfn_train_df = pd.read_csv(_train_fp)
    _tabpfn_valid_df = pd.read_csv(_valid_fp)
    _tabpfn_test_df  = pd.read_csv(_test_fp)

    _tabpfn_clf = tabpfn_model_train(
        train_data=_tabpfn_train_df,
        valid_data=_tabpfn_valid_df,
        tabpfn_model_path="./output/tabpfn_model.pkl",
        y_col_name=y_col_name,
        n_ensembles=8,        # try 4 on small GPUs
        device="auto",        # "cuda" if available else "cpu"
        calibration=False,
    )

    _Xt = (
        _tabpfn_test_df.drop(columns=[y_col_name])
        .select_dtypes(include=[np.number])
        .fillna(0.0)
        .values
    )
    _yt = _tabpfn_test_df[y_col_name].to_numpy()

    _probs = _tabpfn_clf.predict_proba(_Xt)[:, 1]
    _preds = (_probs >= 0.5).astype(int)

    try:
        _auc = roc_auc_score(_yt, _probs)
    except ValueError:
        _auc = float("nan")
    _acc = accuracy_score(_yt, _preds)
    _prec = precision_score(_yt, _preds, zero_division=0)
    _rec = recall_score(_yt, _preds, zero_division=0)
    _f1  = f1_score(_yt, _preds, zero_division=0)

    pd.DataFrame(
        {"AUC":[_auc], "Accuracy":[_acc], "Precision":[_prec], "Recall":[_rec], "F1":[_f1]}
    ).to_csv("./output/evaluation_results_tabpfn_test.csv", index=False)

    plot_and_save_confusion_matrix(_yt, _preds, "./output/confusion_matrix_tabpfn_test.png")
    print("[TabPFN] test metrics saved to ./output/evaluation_results_tabpfn_test.csv")
    print("[TabPFN] test confusion matrix saved to ./output/confusion_matrix_tabpfn_test.png")
    # ===== End TabPFN =====

# Example usage
# main(densenet_train_data, xgb_train_data, valid_data, test_data, y_col_name='PHENO', model_params_densenet=None, model_params_xgb=None)
