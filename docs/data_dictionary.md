# SwiftRoute Logistics — Data Dictionary

---

## shipments.csv
One row per shipment. This is the primary table.

| Column | Type | Description |
|---|---|---|
| `shipment_id` | string | Unique shipment identifier (e.g. `SH0000001`) |
| `pickup_date` | date | Date the parcel was collected from sender |
| `origin_region` | string | UK region where parcel was picked up |
| `dest_region` | string | UK region where parcel is being delivered |
| `service_tier` | string | `express`, `standard`, or `economy` |
| `carrier_id` | string | Which carrier handled this shipment |
| `route_id` | string | Origin–destination route identifier |
| `weight_kg` | float | Parcel weight in kilograms |
| `volume_cm3` | float | Parcel volume in cubic centimetres |
| `declared_value` | float | Declared value of contents (£) |
| `is_business` | int | 1 if sender is a business account, 0 if consumer |
| `is_delayed` | int | **Target variable** — 1 if delivered outside promised window |

---

## routes.csv
One row per unique origin–destination route.

| Column | Type | Description |
|---|---|---|
| `route_id` | string | Unique route identifier |
| `origin_region` | string | Origin UK region |
| `dest_region` | string | Destination UK region |
| `distance_km` | float | Road distance in kilometres |
| `n_depot_hops` | int | Number of intermediate depots on this route |
| `historical_delay_rate` | float | Fraction of shipments on this route that were delayed (last 12 months) |
| `avg_transit_days` | float | Average actual transit time in days |

---

## carriers.csv
One row per carrier.

| Column | Type | Description |
|---|---|---|
| `carrier_id` | string | Unique carrier identifier |
| `carrier_name` | string | Carrier name |
| `tier_coverage` | string | Which service tiers the carrier handles: `all`, `express_only`, `standard_economy` |
| `reliability_score` | float | Composite on-time reliability score (0–1, higher = more reliable) |
| `capacity_utilisation` | float | Current capacity utilisation (0–1); higher = more stretched |
| `avg_delay_mins` | float | Average delay in minutes when a delivery is late |
| `n_active_routes` | int | Number of routes the carrier currently operates |

---

## weather_events.csv
One row per weather disruption event.

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Unique event identifier |
| `region` | string | UK region affected |
| `event_type` | string | `snow`, `flood`, `fog`, `storm`, `high_wind` |
| `severity` | int | 1 (minor) to 3 (severe) |
| `start_date` | date | First day of disruption |
| `end_date` | date | Last day of disruption |

---

## Notes for Students

- A shipment may have **both** its origin and destination region affected by weather — consider how to encode this.
- `weather_events` requires a date-range join: a shipment is weather-affected if its `pickup_date` falls within `[start_date, end_date]` for any event covering its origin or destination region.
- `historical_delay_rate` in `routes.csv` is computed from **past** shipments — it is a valid feature (not leakage), but it lags reality during disruption periods.
- Think about whether `declared_value` could be a proxy for something else (e.g. fragile items, special handling).
