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

    