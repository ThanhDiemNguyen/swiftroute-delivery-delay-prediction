from swiftroute.config import TARGET
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
EXCLUDE = ["shipment_id", "pickup_date", "origin_region", "dest_region", "carrier_id", "route_id", "is_delayed",  "weather_flag", "is_cross_region"]


def split_data(master: pd.DataFrame):
    """Select pickup-time features and create a stratified train/validation split."""

    X = master.drop(columns=EXCLUDE, errors="ignore")
    y = master[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    return X_train, X_val, y_train, y_val

def build_pipeline(model, X: pd.DataFrame, scale: bool = False) -> Pipeline:
    """Impute missing values and one-hot encode categorical features."""
    
    categorical = ["service_tier", "day_of_week"]
    numeric = [col for col in X.columns if col not in categorical]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocess = ColumnTransformer([("numeric", Pipeline(numeric_steps), numeric),("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),])
    
    return Pipeline([("preprocess", preprocess),("model", model),])


def evaluate_model(model: Pipeline, X_train, X_val, y_train, y_val) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_prob = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_prob)

    return {
        "model": model,
        "cv_auc": cv_auc.mean(),
        "val_auc": val_auc,
        "train_auc": train_auc,
        "overfit_gap": train_auc - val_auc,
        "val_prob": val_prob,
    }


def logistic_model(X: pd.DataFrame) -> Pipeline:
    return build_pipeline(
        LogisticRegression(class_weight="balanced", penalty="l2", C=1.0, max_iter=1000),
        X,
        scale=True
    )


def forest_model(X: pd.DataFrame) -> Pipeline:
    return build_pipeline(
        RandomForestClassifier(n_estimators=150,max_depth=10,min_samples_leaf=5,class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=-1,),
        X,
    )


def compare_models(results: dict) -> pd.DataFrame:
    out = pd.DataFrame({})
    for name, result in results.items():
        out.loc[name, "CV AUC"] = result["cv_auc"]
        out.loc[name, "Validation AUC"] = result["val_auc"]
        out.loc[name, "Training AUC"] = result["train_auc"]
        out.loc[name, "Overfit gap"] = result["overfit_gap"]
    return out.round(3)
