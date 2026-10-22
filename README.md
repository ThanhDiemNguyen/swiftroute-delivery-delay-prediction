# SwiftRoute Delivery Delay Prediction

Predicts whether a shipment will be delayed at pickup, so Operations can intervene on the top 15% highest-risk shipments.

## Project structure

```
swiftroute-delivery-delay-prediction/
├── .gitignore
├── .python-version        # Python version used by the project
├── Makefile               # setup and helper commands
├── pyproject.toml         # package configuration
├── requirements.txt       # dependencies
├── README.md
├── data/
│   ├── raw/               # original CSVs: shipments, routes, carriers, weather_events
│   ├── interim/         
│   ├── processed/       
│   └── external/       
├── docs/
│   ├── business_brief.md  # business context
│   └── data_dictionary.md # column definitions
├── models/              
├── notebooks/
│   └── coursework.ipynb   # main coursework notebook
├── references/          
├── reports/
│   └── figures/           # saved plots
└── swiftroute/            # reusable package
    ├── __init__.py
    ├── config.py          # paths and settings
    ├── dataset.py         # load, clean, join
    ├── features.py        # feature engineering
    ├── plots.py           # EDA and diagnostic plots
    ├── extension.py       # route clustering and SHAP
    └── modeling/
        ├── __init__.py
        ├── train.py       # split, models, evaluation
        ├── impact.py      # business impact
        └── predict.py    
```

## Requirements

- macOS with Python 3.10+
- `make` (install with `xcode-select --install` if missing)

## Setup

```bash
cd swiftroute-delivery-delay-prediction
make install
```

`make install` creates the virtual environment and installs the dependencies and the `swiftroute` package.

Place the four CSV files in `data/raw/`.

Or run from the terminal:

## How to run

1. Open `notebooks/coursework.ipynb` in VS Code.
2. Select the kernel created by `make install` (`swiftroute` or the `.venv` interpreter).
3. Choose **Restart**, then **Run All** so all outputs are current.

```bash
source .venv/bin/activate
jupyter notebook notebooks/coursework.ipynb
```

## Outputs

- Figures are saved to `reports/figures/`.
- The trained model is saved as `model.pkl`.
