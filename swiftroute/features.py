import numpy as np
import pandas as pd

def add_features(master: pd.DataFrame) -> pd.DataFrame:
    
    df = master.copy()

    #1. weather_affected: binary flag - any weather at origin OR destination
    # rename weather_flag to weather_affected for clarity
    df = df.rename(columns={"weather_flag": "weather_affected"})

    # 2. weather_both_ends: both origin AND destination affected
    df["weather_both_ends"] = (df["origin_region"].isin(df.loc[df["weather_affected"], "origin_region"]) &
                                df["dest_region"].isin(df.loc[df["weather_affected"], "dest_region"])).astype(int)   

    # 3. is_winter: pickup in Nov-Feb (higher disruption risk)
    df["is_winter"] = df["pickup_date"].dt.month.isin([11, 12, 1, 2]).astype(int)

    # 4. volume_to_weight: proxy for bulky or awkward parcels
    df["volume_to_weight"] = df["volume_cm3"] / df["weight_kg"].replace(0, np.nan)

    # 5. carrier_route_risk: carrier reliability * route historical delay rate
    df["carrier_route_risk"] = (1 - df["reliability_score"]) * df["historical_delay_rate"]

    # 6. day_of_week: Monday/Friday shipments may face different delays
    df["day_of_week"] = df["pickup_date"].dt.dayofweek
    
    # 7. is_cross_region: same vs different region (local vs long-haul)
    df["is_cross_region"] = (df["origin_region"] != df["dest_region"]).astype(int)

    return df