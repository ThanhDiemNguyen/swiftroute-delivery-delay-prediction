
import numpy as np
import pandas as pd


def compute_business_impact(y_val: pd.Series, best_prob: np.ndarray, intervention_rate: float = 0.15) -> pd.DataFrame:
    actual = np.asarray(y_val)
    scores = np.asarray(best_prob)

    if len(actual) != len(scores):
        raise ValueError("y_val and best_prob must have the same length")

    total_delays = int(actual.sum())
    n_flagged = int(np.ceil(len(actual) * intervention_rate))

    # Random 15% policy
    random_delays = total_delays * n_flagged / len(actual)

    # Model top-15% policy
    top_risk = np.argsort(scores)[-n_flagged:]
    model_delays = int(actual[top_risk].sum())

    result = pd.DataFrame({
        "Flagged shipments": [n_flagged, n_flagged],
        "Delays found": [random_delays, model_delays],
    }, index=["Random 15%", "Model top 15%"])

    # Delay capture rate
    result["Capture rate (%)"] = 100 * result["Delays found"] / total_delays

    # SLA penalty avoided
    result["SLA penalty avoided (£)"] = result["Delays found"] * 45

    return result.round(1)
