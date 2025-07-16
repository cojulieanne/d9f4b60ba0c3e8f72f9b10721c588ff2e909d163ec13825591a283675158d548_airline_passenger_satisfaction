import sys
from pathlib import Path
import warnings
import joblib
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(project_root))

from data_preprocessing import *
from model_training import *
from evaluation import *

def main():
    #train, test = get_train_test()

    #data = {
    #    'train': train
    #    ,'test': test
    #}

    #for name, df in data.items():
    #    print(name)
    #    check_uniqueness(df)

    data = get_data()

    check_uniqueness(data)
    data = standardize_cols(data)
    data = impute_missing(data, columns=['arrival_delay_in_minutes'], method = 'median')
    class_map = {'class': {'eco': 1, 'eco plus': 2, 'business': 3}}
    data = label_encode(data, class_map)
    data = one_hot_encode(data)
    data = data.drop(columns=['id'])
    data = split_data(data, 'satisfaction_satisfied')

    get_default_binary_models(data['X_train'], data['y_train'], data['X_test'], data['y_test'])

    best_params, trials = run_xgboost_hyperopt(data['X_train'], data['y_train'], cv = 5, max_evals=50)

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
