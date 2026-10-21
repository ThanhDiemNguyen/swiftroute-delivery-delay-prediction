from swiftroute.config import FIGURES_DIR

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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