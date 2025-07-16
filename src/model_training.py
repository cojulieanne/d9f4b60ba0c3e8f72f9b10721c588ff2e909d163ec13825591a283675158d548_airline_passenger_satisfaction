from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd

from sklearn.model_selection import cross_val_score

from evaluation import evaluate_single_model

def get_default_binary_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a set of default binary classification models.

    Returns:
        dict: model_name → metrics
    """
    models = {
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "lightgbm": LGBMClassifier(verbose = 0)
    }

    print(f"Loaded {len(models)} binary classifiers. Evaluating...")

    results = {}
    for name, model in models.items():
        try:
            __, metrics = evaluate_single_model(model, X_train, y_train, X_test, y_test)
            results[name] = metrics
            print(f"{name} → {metrics}")
        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = "Failed"

    results_df = pd.DataFrame(results).T  # transpose if models are keys
    results_df.to_csv("../results/default_binary_models_results.csv")

    print("Saved results to default_binary_models_results.csv")

    return results


def run_xgboost_hyperopt(X_train, y_train, cv = 5, max_evals=50):
    def objective(params):
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        print(f"Params: {params} | CV AUC: {score:.4f}")

        return {'loss': -score, 'status': STATUS_OK}

    space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'n_estimators': hp.quniform('n_estimators', 50, 300, 25),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    print("\n🔍 Best Parameters Found:")
    print(best)
    return best, trials


def create_xgboost_model(params):
    """
    Trains a final XGBoost model using the full training data and optimized parameters.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        best_params (dict): Optimized hyperparameters from Hyperopt

    Returns:
        XGBClassifier: Trained model
    """

    # Convert int-based parameters
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    # Add default values
    params.update({
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    })

    model = XGBClassifier(**params)
    return model
