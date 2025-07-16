import joblib
import pickle
from data_preprocessing import get_data, check_uniqueness, standardize_cols, impute_missing, label_encode, one_hot_encode, split_data
from model_training import get_default_binary_models, run_xgboost_hyperopt, create_xgboost_model
from evaluation import evaluate_single_model, save_confusion_matrix_plot, save_feature_importance_plot, save_shap_summary_plot, save_roc_curve_plot
import warnings

warnings.filterwarnings('ignore')

def main():

    #Phase 1: Data Preprocessing
    data = get_data()
    check_uniqueness(data)
    data = standardize_cols(data)
    data = impute_missing(data, columns=['arrival_delay_in_minutes'], method = 'median')
    class_map = {'class': {'eco': 1, 'eco plus': 2, 'business': 3}}
    data = label_encode(data, class_map)
    data = one_hot_encode(data)

    #Phase 2: Split Data
    data = data.drop(columns=['id'])
    data = split_data(data, 'satisfaction_satisfied')

    #Phase 3: Model Training
    get_default_binary_models(data['X_train'], data['y_train'], data['X_test'], data['y_test'])

    #Default XGBoost performed best, so it will be the model to tune
    best_params, trials = run_xgboost_hyperopt(data['X_train'], data['y_train'], cv = 5, max_evals=50)
    with open('../results/hyperopt_trials.pkl', 'wb') as f:
        pickle.dump(trials, f)

    #Phase 4: Model Evaluation
    model = create_xgboost_model(best_params)
    model, test_results = evaluate_single_model(model, data['X_train'], data['y_train'], data['X_test'], data['y_test'])
    model_path = '../results/final_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    save_shap_summary_plot(model, data['X_test'])
    save_feature_importance_plot(model, data['X_test'])
    save_roc_curve_plot(model, data['X_test'], data['y_test'])
    save_confusion_matrix_plot(model, data['X_test'], data['y_test'])


if __name__ == "__main__":
    main()
