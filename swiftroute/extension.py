import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap

from swiftroute.config import FIGURES_DIR


def analyze_route_clusters(master: pd.DataFrame) -> pd.DataFrame:
    # One row per route: delay rate, network features and average carrier reliability
    routes = (master.groupby(["route_id", "origin_region", "dest_region"], as_index=False)
            .agg(
                shipments=("is_delayed", "size"),
                delay_rate=("is_delayed", "mean"),
                distance_km=("distance_km", "first"),
                n_depot_hops=("n_depot_hops", "first"),
                historical_delay_rate=("historical_delay_rate", "first"),
                weather_exposure=("weather_affected", "mean"),
                mean_reliability=("reliability_score", "mean"),
            )
    )
    routes["small_sample"] = routes["shipments"] < 30
    

    # Scale features -> no single feature dominates KMeans distances
    columns = ["distance_km", "n_depot_hops", "historical_delay_rate", "weather_exposure", "mean_reliability",]
    scaled = StandardScaler().fit_transform(routes[columns])
    routes["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)

    # Label clusters by average observed delay rate (lowest -> Low risk, highest -> High risk)
    order = routes.groupby("cluster")["delay_rate"].mean().sort_values().index
    routes["risk_group"] = routes["cluster"].map(dict(zip(order, ["Low risk", "Medium risk", "High risk"])))

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=routes, x="distance_km", y="delay_rate",
                    hue="risk_group", hue_order=["Low risk", "Medium risk", "High risk"], ax=ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="Route Risk Clusters", xlabel="Distance (km)", ylabel="Delay rate")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "route_clusters.png", dpi=150)
    plt.show()
    plt.close(fig)
    return routes


def explain_top_shap(best_model, X_train: pd.DataFrame, X_val: pd.DataFrame, master: pd.DataFrame, n: int = 10, borderline: int = 0) -> pd.DataFrame:
    preprocess = best_model.named_steps["preprocess"]
    estimator = best_model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise ValueError("SHAP function expects a fitted Logistic Regression model.")
    if not 1 <= n <= len(X_val) or borderline < 0:
        raise ValueError("Use 1 <= n <= validation size and borderline >= 0.")

    # Pick the highest-risk validation shipments (highest score first)
    scores = best_model.predict_proba(X_val)[:, 1]
    top_positions = np.argsort(scores)[-n:][::-1]
    positions = top_positions.copy()

    if borderline:
        cutoff_index = int(np.ceil(0.15 * len(scores)))
        cutoff = np.sort(scores)[-cutoff_index]
        closest = np.argsort(np.abs(scores - cutoff))
        closest = closest[~np.isin(closest, top_positions)][:borderline]
        positions = np.concatenate([top_positions, closest])

    selected = X_val.iloc[positions]

    # Background sample from training data defines the SHAP baseline
    background = preprocess.transform(X_train.sample(min(500, len(X_train)), random_state=42))
    selected_data = preprocess.transform(selected)
    # SHAP needs dense arrays
    if hasattr(background, "toarray"):
        background = background.toarray()
    if hasattr(selected_data, "toarray"):
        selected_data = selected_data.toarray()

    # Strip the transformer prefix from feature names and compute SHAP values
    names = [s.split("__", 1)[-1] for s in preprocess.get_feature_names_out()]
    explanations = shap.LinearExplainer(estimator, background)(selected_data)
    explanations.feature_names = names
    ids = master.loc[selected.index, "shipment_id"].to_numpy()

    # Per shipment: top 3 features raising and lowering risk
    rows = []
    for i, shipment_id in enumerate(ids):
        effects = pd.Series(explanations.values[i], index=names)
        up = effects[effects > 0].nlargest(3)
        down = effects[effects < 0].nsmallest(3)
        rows.append({
            "shipment_id": shipment_id,
            "group": "Top risk" if i < n else "Near 15% cutoff",
            "risk_score": round(float(scores[positions[i]]), 3),
            "risk_increases": ", ".join(f"{k} ({v:+.2f})" for k, v in up.items()),
            "risk_decreases": ", ".join(f"{k} ({v:+.2f})" for k, v in down.items()),
        })
    summary = pd.DataFrame(rows)

    # Heatmap of SHAP values for the 15 most influential features
    top_values = explanations.values[:n]
    important = np.argsort(np.abs(top_values).mean(axis=0))[-15:][::-1]
    heatmap = pd.DataFrame(top_values[:, important], index=ids[:n], columns=[names[j] for j in important])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap, cmap="RdBu_r", center=0, ax=ax)
    ax.set(title=f"SHAP: {n} Highest-Risk Shipments", xlabel="SHAP contribution (log-odds)", ylabel="Shipment ID")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"shap_top{n}.png", dpi=150)
    plt.show()
    plt.close(fig)


    # Waterfall plot for the single highest-risk shipment
    for i, shipment_id in enumerate(ids):
        shap.plots.waterfall(explanations[i], max_display=12, show=False)
        fig = plt.gcf()
        fig.savefig(FIGURES_DIR / f"shap_{shipment_id}.png", dpi=120, bbox_inches="tight")
        if i == 0 or (borderline and i == n):
            plt.show()
        plt.close(fig)

    return summary