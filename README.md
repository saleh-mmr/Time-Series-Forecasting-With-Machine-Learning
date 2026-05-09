# Campus Load Forecasting Project

## 1. Project Overview

This project focuses on forecasting the electrical load of the Savona Campus using historical load data collected from 2018 to 2023. The original dataset contains measurements at approximately 1-minute resolution. The goal is to clean and prepare the data, explore its behavior, create meaningful forecasting features, and compare different machine learning models for short-term load forecasting.

The final forecasting models were trained using 15-minute average load data. Several models were compared, including a previous-day baseline model, Random Forest, XGBoost, and LightGBM.

---

## 2. Dataset Description

The dataset contains two main columns:

| Column | Description |
|---|---|
| `data_timestamp` | Timestamp of the load measurement |
| `value` | Electrical load value |

The original data was recorded at approximately 1-minute intervals, but the dataset contained several data quality issues, including:

- mixed timestamp formats
- timestamps with milliseconds
- negative load values
- duplicate rows
- duplicate timestamps
- missing timestamps
- long missing periods
- outliers and high-load peaks

Because the project is based on time-series forecasting, careful preprocessing was required before model development.

---

# 3. Data Preprocessing

## 3.1 Importing Required Libraries

The project started by importing the main Python libraries used for data analysis, visualization, preprocessing, and modeling:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `lightgbm`

These libraries were used for reading the dataset, cleaning data, creating features, training models, and evaluating forecasting performance.

---

## 3.2 Loading the Dataset

The dataset was loaded in Google Colab from Google Drive using the shared file link. After loading, the initial structure of the dataset was inspected using:

```python
df.head()
df.tail()
df.info()
df.shape
df.columns
```

This helped confirm the column names, data types, number of records, and general structure of the dataset.

---

## 3.3 Detecting Timestamp Formats

Before converting the timestamp column to datetime, the existing timestamp formats were checked.

Two timestamp formats were found:

| Timestamp Format | Number of Rows |
|---|---:|
| `YYYY-MM-DD HH:MM:SS` | 1,028,996 |
| `YYYY-MM-DD HH:MM:SS.milliseconds` | 1,780,606 |

This showed that the dataset contained mixed timestamp precision. Some timestamps had only seconds, while others included milliseconds.

### Approach

To solve this, the timestamp column was converted using:

```python
pd.to_datetime(
    df['data_timestamp'],
    format='mixed',
    errors='coerce'
)
```

The argument `format='mixed'` allows pandas to parse rows with different datetime formats. The argument `errors='coerce'` converts invalid timestamps to `NaT` instead of stopping the program.

---

## 3.4 Rounding Timestamps to Minute Resolution

After converting timestamps, they were rounded down to exact minute resolution using:

```python
df['data_timestamp'] = df['data_timestamp'].dt.floor('min')
```

### Reason

The original dataset contained timestamps with milliseconds, for example:

```text
2020-01-01 12:30:00.328
```

This caused many false time gaps when comparing consecutive timestamps. By flooring timestamps to the nearest minute, all records were aligned to a regular 1-minute time grid.

This step reduced false gap detections and made the dataset suitable for time-series processing.

---

## 3.5 Handling Negative Load Values

Negative load values were detected in the `value` column.

Since electrical load consumption is normally expected to be non-negative, negative values were considered invalid or suspicious measurements.

### Approach

Instead of deleting rows immediately, negative values were replaced with `NaN`:

```python
df.loc[df['value'] < 0, 'value'] = np.nan
```

### Reason

Replacing negative values with `NaN` preserves the timestamp structure. This is important because deleting rows would create additional artificial gaps in the time series.

The missing values created from negative values were handled later.

---

## 3.6 Handling Missing Values in the Load Column

After replacing negative values with `NaN`, missing values in the load column were handled using interpolation.

### Approach

Linear interpolation was applied:

```python
df['value'] = df['value'].interpolate(method='linear')
```

### Reason

For short missing sequences, load values usually change smoothly over time. Linear interpolation is a simple and reasonable method for estimating missing values between two known points.

However, interpolation was not used blindly for long missing timestamp gaps, because this could create unrealistic artificial data.

---

## 3.7 Handling Duplicate Rows and Duplicate Timestamps

Duplicates were checked in two ways:

1. Fully duplicated rows
2. Duplicate timestamps

### Fully Duplicate Rows

Fully duplicated rows were removed using:

```python
df = df.drop_duplicates()
```

### Duplicate Timestamps

Duplicate timestamps were removed using:

```python
df = df.drop_duplicates(
    subset='data_timestamp',
    keep='first'
)
```

### Reason

In time-series forecasting, each timestamp should correspond to one observation. Duplicate timestamps can create problems during:

- resampling
- lag feature generation
- rolling feature calculation
- train/test splitting

The first occurrence was kept for each duplicated timestamp.

---

## 3.8 Identifying Missing Timestamps

After cleaning timestamp precision and duplicates, missing timestamps were identified.

The dataset was sorted by timestamp, and the time difference between consecutive rows was calculated:

```python
df['time_diff'] = df['data_timestamp'].diff()
```

Gap events were detected where the time difference was larger than 1 minute:

```python
gap_events = df[df['time_diff'] > pd.Timedelta(minutes=1)]
```

### Findings

After rounding timestamps, the dataset contained:

- 4,721 gap events larger than 1 minute
- 179,468 missing 1-minute timestamps

The largest gaps included:

| Gap Duration |
|---:|
| 61 days |
| 15 days |
| 13 days |

### Interpretation

The 4,721 value represents the number of separate gap events, while 179,468 represents the total number of missing 1-minute timestamps.

For example, if the data jumps from `10:01` to `10:10`, this is one gap event, but eight individual timestamps are missing.

---

## 3.9 Handling Missing Timestamps

A complete 1-minute time index was created using:

```python
full_time_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq='1min'
)
```

Then the dataset was reindexed to this complete timeline:

```python
df = df.reindex(full_time_index)
```

This inserted missing timestamps into the dataset. The missing load values appeared as `NaN`.

### Result

After reindexing:

- Dataset length became 2,988,584 rows
- 179,468 missing timestamps were inserted
- These inserted timestamps had missing load values

---

## 3.10 Filling Night-Time Missing Gaps

The project used a specific strategy to fill missing values only during night-time hours, from 23:00 to 07:00.

### Reason

Campus load is usually more stable during the night because there is less human activity and lower operational variation. Therefore, filling missing night-time values is more acceptable than filling daytime gaps.

A night mask was created:

```python
night_mask = (df.index.hour >= 23) | (df.index.hour < 7)
```

Missing night-time values were filled using forward fill:

```python
df.loc[night_mask & missing_mask, 'value'] = df['value'].ffill()[night_mask & missing_mask]
```

A safer version can include a limit:

```python
df['value'].ffill(limit=480)
```

This avoids propagating one old value across very long gaps.

### Findings

The night-time filling step produced:

| Item | Value |
|---|---:|
| NaN values before night fill | 170,103 |
| NaN values after night fill | 166,943 |
| NaN values filled | 3,160 |

### Interpretation

Only a small portion of missing values occurred during fillable night-time periods. Most remaining missing values belonged to long gaps.

---

## 3.11 Removing Remaining Missing Values

After night-time filling, many missing values still remained. These were mainly caused by long missing periods, such as gaps of several days or weeks.

### Approach

The remaining missing values were removed before modeling:

```python
df_clean = df.dropna(subset=['value']).copy()
```

### Reason

Long gaps should not be filled using simple interpolation or forward filling because this would create artificial load patterns. Removing them is safer and more scientifically defensible for model training.

---

## 3.12 Resampling to 15-Minute Resolution

The original data was at 1-minute resolution. For forecasting, it was resampled to 15-minute average load:

```python
df_15min = df_clean.resample('15min').mean()
```

### Reason

The project description stated that the 1-minute data can be averaged to 15 minutes. Resampling reduces noise and makes the dataset more manageable for forecasting.

### Result

After resampling, the cleaned 15-minute dataset had:

| Metric | Value |
|---|---:|
| Start date | 2017-12-31 23:00 |
| End date | 2023-09-07 08:30 |
| Mean load | 109.49 |
| Standard deviation | 57.26 |
| Minimum load | 1.23 |
| Maximum load | 406.74 |

After resampling, remaining missing 15-minute rows were removed:

```python
df_15min_clean = df_15min.dropna(subset=['value']).copy()
```

---

# 4. Data Visualization and Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and behavior of the load data before modeling.

---

## 4.1 Full Load Time-Series Plot

The cleaned 15-minute load series was plotted over the full time range.

### Purpose

This visualization helped identify:

- long-term trends
- abnormal periods
- seasonality
- general load behavior over time

---

## 4.2 One-Week Load Plot

A one-week period was plotted to observe short-term patterns more clearly.

### Purpose

The full time series is very dense, so a one-week plot helps show:

- daily cycles
- differences between days
- peak and off-peak periods
- short-term variability

---

## 4.3 Average Daily Load Profile

The average daily profile was calculated by grouping data by time of day.

```python
daily_profile = eda_df.groupby('time_of_day')['value'].mean()
```

### Purpose

This showed how load changes over a typical 24-hour day.

### Expected Insight

Campus load is usually lower during night hours and higher during daytime working hours.

---

## 4.4 Average Weekly Load Profile

The weekly profile was calculated by grouping the data by day of the week.

```python
weekly_profile = eda_df.groupby('day_name')['value'].mean()
```

### Purpose

This helped compare weekdays and weekends.

### Expected Insight

Weekdays usually have higher average load due to academic and operational activity, while weekends tend to have lower load.

---

## 4.5 Average Monthly Load Profile

The monthly profile was calculated by grouping data by month.

```python
monthly_profile = eda_df.groupby('month_name')['value'].mean()
```

### Purpose

This helped identify seasonal changes in electricity consumption.

### Possible Causes of Seasonality

Monthly differences may be caused by:

- heating demand
- cooling demand
- academic calendar
- holidays
- seasonal campus activity

---

## 4.6 Load Distribution

A histogram of the 15-minute load values was plotted.

### Purpose

This helped understand:

- the most common load range
- skewness of the distribution
- frequency of very high or very low values
- possible abnormal values

---

# 5. Outlier Detection and Treatment

## 5.1 IQR-Based Outlier Detection

Outliers were detected using the Interquartile Range method.

The quartiles were:

| Statistic | Value |
|---|---:|
| Q1 | 69.45 |
| Q3 | 129.06 |
| IQR | 59.61 |
| Lower bound | -19.97 |
| Upper bound | 218.48 |

Values above the upper bound were classified as high-load outliers.

### Findings

| Item | Value |
|---|---:|
| Number of outliers | 13,594 |
| Percentage of dataset | 7.23% |

---

## 5.2 Interpretation of Outliers

The lower bound was negative, but the cleaned dataset contained only positive load values, so low-load outliers were not a major concern.

The main outliers were high-load values above approximately 218.48.

However, 7.23% of the data was classified as outliers. This is a relatively large proportion, suggesting that many of these values may be real high-load events rather than data errors.

---

## 5.3 Visualizing Outliers Over Time

Outliers were plotted over the time series to understand when they occurred.

### Purpose

This helped determine whether outliers were random errors or structured events.

If outliers are concentrated in specific seasons or months, they are more likely to represent real high-load behavior, such as heating or cooling demand.

---

## 5.4 Outliers by Month

The number of outliers was counted for each month.

### Purpose

This helped understand whether high-load outliers were seasonal.

### Decision

The outliers were kept in the dataset.

### Reason

Since the outliers represented a significant percentage of the data and may correspond to real high-demand periods, removing them could harm the forecasting model. High load peaks are important for load forecasting, so they should be preserved unless there is strong evidence that they are measurement errors.

---

# 6. Feature Engineering

Feature engineering was performed after cleaning and resampling the dataset to 15-minute intervals.

The final modeling dataframe was created from:

```python
model_df = df_15min_clean.copy()
```

---

## 6.1 Time-Based Features

The following calendar features were extracted from the timestamp index:

| Feature | Description |
|---|---|
| `hour` | Hour of the day |
| `minute` | Minute of the hour |
| `day_of_week` | Day of week, Monday = 0 and Sunday = 6 |
| `day_of_month` | Day of the month |
| `month` | Month number |
| `year` | Year |
| `is_weekend` | Binary feature: 1 for Saturday/Sunday, 0 otherwise |

### Reason

Load consumption depends strongly on time. Campus activity usually follows daily, weekly, and seasonal patterns.

These features help the model learn:

- daily cycles
- working hours
- weekend behavior
- monthly and seasonal effects
- long-term yearly variation

---

## 6.2 Lag Features

Lag features were created from previous load values.

Because the dataset was resampled to 15-minute intervals:

| Time Duration | Number of 15-Min Steps |
|---|---:|
| 15 minutes | 1 |
| 1 hour | 4 |
| 1 day | 96 |
| 1 week | 672 |

The following lag features were created:

| Feature | Description |
|---|---|
| `lag_1` | Load from previous 15 minutes |
| `lag_4` | Load from previous 1 hour |
| `lag_96` | Load from same time previous day |
| `lag_672` | Load from same time previous week |

### Reason

Electrical load is highly autocorrelated. Current load is usually related to recent load, the load at the same time yesterday, and the load at the same time last week.

Lag features are therefore essential for time-series forecasting.

---

## 6.3 Rolling Mean Features

Rolling average features were created using previous values only.

The following rolling features were used:

| Feature | Description |
|---|---|
| `rolling_mean_4` | Average load over the previous 1 hour |
| `rolling_mean_96` | Average load over the previous 1 day |
| `rolling_mean_672` | Average load over the previous 1 week |

The rolling features were calculated using:

```python
model_df['value'].shift(1).rolling(window=...).mean()
```

### Reason

The `shift(1)` is important because it prevents data leakage. Without shifting, the rolling average would include the current target value, which would give the model information it would not have in a real forecasting scenario.

Rolling averages help the model understand recent load trends and smooth behavior.

---

## 6.4 Rolling Standard Deviation Feature

A rolling standard deviation feature was created:

| Feature | Description |
|---|---|
| `rolling_std_96` | Standard deviation of the previous 1 day |

### Reason

This feature captures recent variability in the load. Some periods may be stable, while others may be more volatile. This information can help the model adjust its predictions.

---

## 6.5 Removing Rows with Missing Feature Values

Lag and rolling features create missing values at the beginning of the dataset because earlier historical values are not available.

These rows were removed using:

```python
model_df = model_df.dropna().copy()
```

### Reason

Machine learning models require complete input features. Removing these initial rows is standard practice after creating lag and rolling features.

---

# 7. Modeling Summary

After preprocessing, visualization, outlier analysis, and feature engineering, the dataset was split using a time-based train/test split.

An 80/20 split was used:

- first 80% of the time series for training
- last 20% for testing

Random splitting was not used because it would mix past and future data and cause data leakage.

The following models were evaluated:

1. Previous-day baseline
2. Random Forest
3. XGBoost
4. LightGBM

## Model Results

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Previous Day Baseline | 22.29 | 40.25 | 22.29% |
| Random Forest | 5.21 | 7.91 | 5.14% |
| XGBoost | 5.10 | 7.75 | 4.99% |
| LightGBM | 5.07 | 7.70 | 4.97% |

## Best Model

The best model was LightGBM, with:

| Metric | Value |
|---|---:|
| MAE | 5.07 |
| RMSE | 7.70 |
| MAPE | 4.97% |

LightGBM slightly outperformed XGBoost and Random Forest, while all machine learning models significantly outperformed the previous-day baseline.

---

# 8. Final Preprocessing and Modeling Decisions

The main decisions made in this project were:

| Issue | Approach |
|---|---|
| Mixed timestamp formats | Parsed using `format='mixed'` |
| Milliseconds in timestamps | Rounded down to minute resolution |
| Negative load values | Replaced with `NaN` |
| Missing load values | Interpolated when appropriate |
| Duplicate rows | Removed |
| Duplicate timestamps | First occurrence kept |
| Missing timestamps | Complete 1-minute index created |
| Night-time gaps | Filled between 23:00 and 07:00 |
| Long gaps | Not fully interpolated; remaining NaNs removed |
| 1-minute data | Resampled to 15-minute average |
| Outliers | Detected using IQR but kept |
| Forecasting features | Time, lag, rolling mean, rolling standard deviation |
| Train/test split | Time-based 80/20 split |

---

# 9. Conclusion

This project developed a complete machine-learning pipeline for campus load forecasting. The dataset required significant preprocessing because it contained mixed timestamp formats, duplicates, negative values, missing timestamps, and long missing periods.

After cleaning and resampling the data to 15-minute intervals, several exploratory analyses were performed to understand daily, weekly, and monthly load behavior. Outliers were detected using the IQR method, but they were kept because they likely represented real high-load events rather than measurement errors.

Feature engineering was a key part of the project. Time-based features, lag features, and rolling statistics allowed the models to learn daily, weekly, and seasonal patterns.

The final results showed that machine learning models greatly improved forecasting performance compared with the previous-day baseline. LightGBM achieved the best result, with a MAPE of approximately 4.97%.

This confirms that machine-learning models, combined with careful preprocessing and time-series feature engineering, can effectively forecast campus electrical load.
