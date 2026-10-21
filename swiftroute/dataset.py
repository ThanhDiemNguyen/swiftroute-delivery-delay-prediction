import pandas as pd

from swiftroute.config import RAW_DIR, TARGET


def load_dataset() -> dict[str, pd.DataFrame]:
    """Load the four raw CSV files into a dict of DataFrames."""
    return {
        "shipments": pd.read_csv(RAW_DIR / "shipments.csv", parse_dates=["pickup_date"]),
        "routes": pd.read_csv(RAW_DIR / "routes.csv"),
        "carriers": pd.read_csv(RAW_DIR / "carriers.csv"),
        "weather": pd.read_csv(RAW_DIR / "weather_events.csv", parse_dates=["start_date", "end_date"]),
    }


def check_dataset(data: dict[str, pd.DataFrame]) -> None:
    """Check data quality for all datasets."""

    for name, df in data.items():
        print(f"\n--- {name.upper()} {df.shape} ---")

        # Data types and null counts
        summary = pd.DataFrame({
            "Dtype": df.dtypes,
            "Missing": df.isna().sum(),
            "Missing (%)": (df.isna().mean() * 100).round(2),
        })
        print(summary.to_string())

        # Duplicates
        print(f"Duplicate rows: {df.duplicated().sum()}")

    shipments = data["shipments"]

    # Referential integrity
    print("\n--- REFERENTIAL INTEGRITY ---")
    print("Orphan carrier_id:", (~shipments["carrier_id"].isin(data["carriers"]["carrier_id"])).sum())
    print("Orphan route_id:", (~shipments["route_id"].isin(data["routes"]["route_id"])).sum())
    print("Duplicate shipment_id:", shipments["shipment_id"].duplicated().sum())

    # Overall delay rate
    print("\n--- TARGET DISTRIBUTION ---")
    print(f"Total shipments: {len(shipments):,}")
    print(f"Delayed shipments: {shipments[TARGET].sum():,}")
    print(f"Overall delay rate: {shipments[TARGET].mean():.2%}")


def clean_dataset(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cleaned = data.copy()
    # Remove duplicate shipments
    cleaned["shipments"] = cleaned["shipments"].drop_duplicates(subset="shipment_id", keep="first").copy()
    return cleaned


def join_routes_carriers(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    shipments, routes, carriers = data["shipments"], data["routes"], data["carriers"]
    route_cols = ["route_id", "distance_km", "n_depot_hops", "historical_delay_rate", "avg_transit_days"]
    carrier_cols = ["carrier_id", "reliability_score", "capacity_utilisation"]

    # Join shipments with route and carrier features
    df = shipments.merge(routes[route_cols], on="route_id", how="left", validate="many_to_one")
    df = df.merge(carriers[carrier_cols], on="carrier_id", how="left", validate="many_to_one")

    return df


def join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    matches = []

    # Check weather events at origin and destination
    for region in ["origin_region", "dest_region"]:
        hit = df[["shipment_id", "pickup_date", region]].merge(weather, left_on=region, right_on="region")

        # Keep events active on pickup date
        hit = hit[hit["pickup_date"].between(hit["start_date"], hit["end_date"])]

        matches.append(hit[["shipment_id", "severity"]])

    # Maximum severity across matching events
    severity = (pd.concat(matches).groupby("shipment_id")["severity"].max())

    df = df.copy()
    df["weather_severity"] = (df["shipment_id"].map(severity).fillna(0).astype(int))
    df["weather_flag"] = df["weather_severity"] > 0

    return df

