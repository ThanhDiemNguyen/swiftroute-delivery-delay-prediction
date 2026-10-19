# SwiftRoute Logistics — Delivery Delay Prediction
## Mini Coursework Business Brief

---

### Company Background

**SwiftRoute Logistics** is a UK-based parcel delivery company serving ~2,400 business clients across England, Scotland, and Wales. The company operates a hub-and-spoke network with 14 regional depots, and handles approximately 85,000 shipments per month across three service tiers:

- **Express** — next-day delivery, guaranteed by 12:00
- **Standard** — 2–3 business day delivery
- **Economy** — 3–5 business day delivery

On-time delivery performance is SwiftRoute's primary competitive differentiator. The company's current SLA (Service Level Agreement) promises a **98% on-time rate** for Express and **95%** for Standard. In 2025, actual performance has fallen to **91% Express / 88% Standard** — triggering SLA penalty payments to clients and threatening contract renewals.

---

### The Business Problem

SwiftRoute's Operations team currently has no early warning system for shipments likely to be delayed. By the time a delay is identified, the shipment is already at the wrong depot or stuck in a weather event — too late to intervene.

The team wants a model that scores each shipment **at the point of pickup** — before it enters the network — and flags the highest-risk 15% for proactive intervention (e.g. re-routing to a faster carrier, pre-alerting the recipient, or upgrading the service tier at cost).

---

### Your Task

You have been given access to SwiftRoute's historical shipment data. Your task is to:

1. Explore and understand the data
2. Build a machine learning model that predicts whether a shipment will be delayed
3. Evaluate the model and assess its business value
4. Save the model for deployment

**Snapshot**: The dataset covers shipments picked up between 2024-01-01 and 2026-06-20. The label `is_delayed = 1` means the shipment was delivered later than the promised window.

---

### Data Available

| File | Description |
|---|---|
| `shipments.csv` | One row per shipment — origin, destination, service tier, weight, dimensions |
| `routes.csv` | One row per origin–destination route — historical on-time rate, distance |
| `carriers.csv` | One row per carrier — reliability scores, capacity utilisation |
| `weather_events.csv` | One row per weather disruption event — region, severity, date range |

---

### Success Criteria

- **Primary metric**: AUC-ROC. Above **0.75** is usable; above **0.80** is strong for this problem.
- **Business constraint**: Flag the top **15%** highest-risk shipments for intervention. At this rate, the model should identify a disproportionate share of actual delays — measured by the **delay capture rate** (% of all delays found in the top 15%).
- **Deployment**: Save your best model as `model.pkl` using `joblib`.

---

### Guidance

- The `weather_events` table covers disruptions by region and date — you must join it to shipments by matching the shipment's origin/destination region and pickup date.
- Some routes and carriers have sparse history — handle missing values carefully.
- Delay rates differ significantly by service tier, carrier, and route — your EDA should surface these patterns.
- Think carefully about which features might cause **data leakage** (information only available after a delay has already occurred).
