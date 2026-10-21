from swiftroute.config import FIGURES_DIR

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve


def plot_key_distributions(data: dict[str, pd.DataFrame]) -> None:
    shipments, carriers, routes = data["shipments"], data["carriers"], data["routes"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # 1. Service tier mix
    order = shipments["service_tier"].value_counts().index
    sns.countplot(data=shipments, x="service_tier", order=order, ax=axes[0, 0])
    axes[0, 0].bar_label(axes[0, 0].containers[0])
    axes[0, 0].set_title("Service tier mix")

    # 2. Weight distribution
    sns.histplot(shipments["weight_kg"].dropna(), bins=50, log_scale=True, ax=axes[0, 1])
    axes[0, 1].set_title("Weight distribution")

    # 3. Delay rate over time (monthly)
    monthly = shipments.set_index("pickup_date")["is_delayed"].resample("MS").mean() * 100
    monthly.plot(ax=axes[1, 0], marker="o", label="Monthly")
    axes[1, 0].axhline(shipments["is_delayed"].mean() * 100, color="red", ls="--", label="Overall")
    axes[1, 0].set_title("Delay rate over time (monthly)")
    axes[1, 0].set_ylabel("Delay Rate (%)")
    axes[1, 0].legend()

    # 4. Carrier reliability scores
    sns.barplot(data=carriers.sort_values("reliability_score"),
                x="reliability_score", y="carrier_id", ax=axes[1, 1])
    axes[1, 1].set_title("Carrier reliability scores")

    # 5. Route distance distribution
    sns.histplot(routes["distance_km"], bins=20, ax=axes[2, 0])
    axes[2, 0].set_title("Route distance distribution")

    # 6. Empty plot for spacing
    axes[2, 1].axis("off")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "key_distributions.png", dpi=150)
    plt.show()

def plot_delay_factors(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    shipments, routes, carriers, weather = data["shipments"], data["routes"], data["carriers"], data["weather"]
    df = shipments.merge(routes[["route_id", "distance_km"]], on="route_id").merge(carriers[["carrier_id", "reliability_score"]], on="carrier_id")
    
    # Weather flag: any event in origin or destination region on the pickup date
    def check_weather(region_col: str) -> pd.Series:
        hit = df[["shipment_id", region_col, "pickup_date"]].merge(weather, left_on=region_col, right_on="region")
        hit = hit[(hit["start_date"] <= hit["pickup_date"]) &(hit["pickup_date"] <= hit["end_date"])]
        return df["shipment_id"].isin(hit["shipment_id"])

    df["weather_flag"] = check_weather("origin_region") | check_weather("dest_region")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Service tier
    sns.barplot(data=df, x="service_tier", y="is_delayed", errorbar=None, ax=axes[0, 0])
    axes[0, 0].set_title("Delay Rate by Service Tier")

    # Carrier reliability (one point per carrier)
    c = df.groupby("carrier_id").agg(rate=("is_delayed", "mean"), rel=("reliability_score", "first"))
    sns.regplot(data=c, x="rel", y="rate", ax=axes[0, 1])
    axes[0, 1].set_title("Delay Rate vs Carrier Reliability")

    # Route distance
    df["distance_group"] = pd.qcut(df["distance_km"], 5)
    sns.barplot(data=df, x="distance_group", y="is_delayed", errorbar=None, ax=axes[1, 0])
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set_title("Delay Rate by Route Distance")

    # Weather
    sns.barplot(data=df, x="weather_flag", y="is_delayed", errorbar=None, ax=axes[1, 1])
    axes[1, 1].set_title("Delay Rate: Weather vs No Weather")
    
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Delay Rate (%)")
    
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "delay_factors.png", dpi=150)
    plt.show()

    return df


def plot_risky_routes_carriers(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    shipments, routes = data["shipments"], data["routes"]

    # Delay rate and number of shipments per route
    route_risk = (
        shipments.groupby(["route_id", "origin_region", "dest_region"], as_index=False)
        .agg(delay_rate=("is_delayed", "mean"), n=("is_delayed", "size"))
    )
    route_risk["route"] = route_risk["origin_region"] + " → " + route_risk["dest_region"] + " (" + route_risk["route_id"].astype(str) + ")"
    top10 = route_risk[route_risk["n"] >= 30].nlargest(10, "delay_rate")

    # Delay rate by carrier
    carrier_risk = (shipments.groupby("carrier_id")["is_delayed"].mean().sort_values(ascending=False).reset_index(name="delay_rate"))

    # Delay rate by origin and destination
    pivot = shipments.pivot_table(index="origin_region", columns="dest_region", values="is_delayed", aggfunc="mean")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Top 10 highest-delay routes
    sns.barplot(data=top10, x="delay_rate", y="route", errorbar=None, ax=axes[0])
    axes[0].set_title("Top 10 Highest-Delay Routes (n >= 30)")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_xlabel("Delay Rate (%)")

    # Delay rate by carrier
    sns.barplot(data=carrier_risk, x="carrier_id", y="delay_rate", errorbar=None, ax=axes[1])
    axes[1].set_title("Delay Rate by Carrier")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylabel("Delay Rate (%)")

    # Origin x destination heatmap
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds", ax=axes[2])
    axes[2].set_title("Delay Rate by Origin and Destination")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "risky_routes_carriers.png", dpi=150)
    plt.show()

    return route_risk


def plot_model_diagnostics(results: dict, y_val: pd.Series, best_name: str) -> None:
    best_model = results[best_name]["model"]
    best_prob = results[best_name]["val_prob"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1. ROC curves for all models
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_val, result["val_prob"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={result['val_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("FP Rate")
    axes[0].set_ylabel("TP Rate")
    axes[0].legend()

    # 2. Score distribution by delay=0 vs delay=1
    for actual, label in [(0, "On time"), (1, "Delayed")]:
        axes[1].hist(best_prob[y_val.to_numpy() == actual], bins=25, density=True, alpha=0.6, label=label)

    axes[1].set_title(f"Score distribution by delay=0 vs delay=1 - {best_name}")
    axes[1].set_xlabel("Predicted Delay Probability")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # 3. Feature importance / coefficients (top 15)
    names = best_model.named_steps["preprocess"].get_feature_names_out()
    names = [name.split("__", 1)[-1] for name in names]

    model = best_model.named_steps["model"]

    if hasattr(model, "coef_"):
        importance = pd.Series(model.coef_[0], index=names)
        title = "Coefficients (top 15)"
    else:
        importance = pd.Series(model.feature_importances_, index=names)
        title = "Feature importances (top 15)"

    top15 = importance.loc[importance.abs().nlargest(15).index].sort_values()

    top15.plot(kind="barh", ax=axes[2])
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_title(f"{title} - {best_name}")
    axes[2].set_xlabel("Coefficient" if hasattr(model, "coef_") else "Importance")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "model_diagnostics.png", dpi=150)
    plt.show()
