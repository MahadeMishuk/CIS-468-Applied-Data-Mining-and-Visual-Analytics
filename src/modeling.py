# src/modeling.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def split_regression(df_ml: pd.DataFrame, target_col="Calories (kcal)", test_size=0.2, random_state=42):
    y = df_ml[target_col]
    X = df_ml.drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_baseline_regressors(X_train, y_train, X_test, y_test):
    """
    Train Linear Regression, Random Forest, and XGBoost (baseline).
    Returns predictions + metrics dataframe.
    """
    results = []
    preds = {}

    # Linear Regression
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    pred_lin = linreg.predict(X_test)
    preds["Linear Regression (baseline)"] = pred_lin
    results.append(_reg_metrics("Linear Regression (baseline)", y_test, pred_lin))

    # Random Forest (baseline)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    preds["Random Forest (baseline)"] = pred_rf
    results.append(_reg_metrics("Random Forest (baseline)", y_test, pred_rf))

    # XGBoost (baseline)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    preds["XGBoost (baseline)"] = pred_xgb
    results.append(_reg_metrics("XGBoost (baseline)", y_test, pred_xgb))

    metrics_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return {"linreg": linreg, "rf": rf, "xgb": xgb}, preds, metrics_df


def tune_random_forest(X_train, y_train, X_test, y_test):
    param_dist = {
        "n_estimators": [150, 250, 400, 600],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base = RandomForestRegressor(random_state=42)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    metrics = _reg_metrics("Random Forest (tuned)", y_test, pred)
    return model, pred, metrics, search.best_params_


def tune_xgboost(X_train, y_train, X_test, y_test):
    param_dist = {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [1, 5, 10],
        "gamma": [0, 1, 5],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    metrics = _reg_metrics("XGBoost (tuned)", y_test, pred)
    return model, pred, metrics, search.best_params_


def build_stacking_ensemble(X_train, y_train, X_test, y_test, rf_tuned, xgb_tuned):
    base_lin = LinearRegression()
    base_rf = RandomForestRegressor(**rf_tuned.get_params())
    base_xgb = XGBRegressor(**xgb_tuned.get_params())

    estimators = [
        ("lin", base_lin),
        ("rf", base_rf),
        ("xgb", base_xgb),
    ]

    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        n_jobs=-1,
    )

    stack_reg.fit(X_train, y_train)
    pred_stack = stack_reg.predict(X_test)
    metrics = _reg_metrics("Stacking Ensemble", y_test, pred_stack)
    return stack_reg, pred_stack, metrics



# Neural Network
def build_and_train_nn(X_train, y_train, validation_split=0.2, epochs=300, batch_size=32):
    input_dim = X_train.shape[1]

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae", "mse"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    return model, history


def evaluate_nn(model, X_test, y_test):
    y_pred = model.predict(X_test).ravel()
    return y_pred, _reg_metrics("Neural Network (Keras)", y_test, y_pred)



#Classification utilities

def split_classification(df_ml: pd.DataFrame, label_col="Healthy", test_size=0.2, random_state=42):
    y = df_ml[label_col]
    X = df_ml.drop(columns=[label_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_classifiers(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression and Decision Tree, return metrics + confusion matrices.
    """
    reports = {}

    # Logistic Regression
    log_clf = LogisticRegression(max_iter=1000)
    log_clf.fit(X_train, y_train)
    pred_log = log_clf.predict(X_test)
    reports["logistic"] = {
        "model": log_clf,
        "accuracy": accuracy_score(y_test, pred_log),
        "report": classification_report(y_test, pred_log, output_dict=False),
        "cm": confusion_matrix(y_test, pred_log),
    }

    # Decision Tree
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_clf.fit(X_train, y_train)
    pred_tree = tree_clf.predict(X_test)
    reports["tree"] = {
        "model": tree_clf,
        "accuracy": accuracy_score(y_test, pred_tree),
        "report": classification_report(y_test, pred_tree, output_dict=False),
        "cm": confusion_matrix(y_test, pred_tree),
    }

    return reports



def _reg_metrics(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }
