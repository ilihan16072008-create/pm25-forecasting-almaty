# PM2.5 Forecasting for Almaty

Can machine learning predict tomorrow's air pollution in Almaty?
I built a Telegram bot that shows current air quality, but it could not tell users what tomorrow would look like. This project tests whether machine learning can fix that.

## Research Question

To what extent can machine learning models improve PM2.5 forecasting in Almaty compared to simple baseline methods?

## Results

| Model | MAE (µg/m³) |
|---|---|
| Persistence (baseline) | 7.83 |
| **Moving Average, 24h (baseline)** | **4.09** |
| Seasonal Average (baseline) | 45.69 |
| Ridge Regression | 8.04 |
| Random Forest | 7.85 |
| XGBoost | 7.43 |

The simple 24-hour moving average beat every machine learning model.

Reason: the training data is mostly winter (high PM2.5 from heating), the test data is summer (low PM2.5). Models learned winter patterns and overestimated summer values.

Takeaway: a more complex model is not automatically a better one.

## Data

 OpenAQ archive — station run by Almaty Air Initiative, AirGradient sensor, CC BY 4.0
 
 IQAir API — hourly collection via GitHub Actions                                    
 
 Period: 23 December 2025 – 4 July 2026                                              
 ~4,600 hourly records (96 hours missing, 2.1%)                                      

 Variables: PM2.5, PM0.3, temperature, humidity                                      

## Files

```
.github/workflows/
  collect_data.yml        hourly data collection

data/
  air_quality_history.csv live data
  dataopenaq_part*.csv    raw OpenAQ exports
  openaq_merged.csv       merged dataset
  features.csv            features + target
  eda_plots/              figures

scripts/
  collect_data.py         fetch hourly data
  merge_openaq_data.py    merge archive parts
  eda_analysis.py         missing-data check + plots
  build_features.py       lags, rolling means, calendar
  baseline_models.py      three baselines
  model_ridge.py          Ridge Regression
  model_random_forest.py  Random Forest
  model_xgboost.py        XGBoost
  model_plots.py          predicted vs actual plots
```

## How to Run

```bash
pip install pandas matplotlib scikit-learn xgboost requests
```

```bash
python scripts/merge_openaq_data.py
python scripts/eda_analysis.py
python scripts/build_features.py
python scripts/baseline_models.py
python scripts/model_ridge.py
python scripts/model_random_forest.py
python scripts/model_xgboost.py
python scripts/model_plots.py
```

## Method Notes

 Data split by time (70% train / 15% validation / 15% test), never shuffled — it is a time series

 Rolling averages shifted by one hour so the current value is excluded (no data leakage)

 Target: mean PM2.5 over the next 24 hours

## Limitations
 
 Only 7 months of data — no full year
 
 One monitoring station
 
 No weather forecasts used as input
 
## Paper

The full research paper: [PM2.5ALA_paper.pdf](PM2.5ALA_paper.pdf)

## Author
Ilikhan Mussayev, Grade 11, Gymnasium No. 15, Almaty, Kazakhstan

## Credits

Air quality data from [OpenAQ](https://openaq.org/) and Almaty Air Initiative, CC BY 4.0.

This was my first project of this kind, and I learned most of it as I went. I used Claude (Anthropic) as a tutor and a helper: it explained methods and concepts I did not know yet, helped me write and debug parts of the code, checked my English grammar, and helped me find papers to read. I made the decisions, ran everything myself, and wrote the research paper text on my own.

