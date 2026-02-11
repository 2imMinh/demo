# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

# Defaults
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# Load the data
data = pd.read_csv('coffee_price.csv', engine='python')
# A bit of pre-processing to make it nicer
data['Ngày']=pd.to_datetime(data['Ngày'], format='%d/%m/%Y')
data.set_index(['Ngày'], inplace=True)
data.sort_index(inplace=True)

# Convert column 2 ("Lần cuối") from string to numeric and plot it
data['Lần cuối'] = data['Lần cuối'].astype(str).str.replace(',', '', regex=False)
data['Lần cuối'] = pd.to_numeric(data['Lần cuối'], errors='coerce')

# Plot only the "Lần cuối" (last price)
data['Lần cuối'].plot()
plt.ylabel('Daily coffee price (USD)')
plt.xlabel('Date')
plt.title('Lần cuối (Last price)')
plt.savefig('results/plot.png')
plt.show()

# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 1)
# Define the p parameters to take any value between 0 and 3
p = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
# Print up to the first few available combinations (safe against short lists)
max_examples = min(4, len(pdq), len(seasonal_pdq))
for i in range(max_examples):
    print(f'SARIMAX: {pdq[i]} x {seasonal_pdq[i]}')

# Helper to parse date strings (accepts dd/mm/YYYY or ISO formats)
def _parse_date(s):
    dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors='coerce')
    return dt

train_data = data.loc[_parse_date('14/01/2008'):_parse_date('28/01/2025')]
test_data = data.loc[_parse_date('28/01/2025'):_parse_date('28/01/2026')]

print(f"Train data shape: {train_data.shape}")
print(f"Train data date range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Test data shape: {test_data.shape}")

warnings.filterwarnings("ignore") # specify to ignore warning messages

# Prepare a clean numeric series for SARIMAX: cast to float and drop NaNs
train_series = train_data['Lần cuối'].astype(float).dropna()
if train_series.empty:
    print("Error: training series 'Lần cuối' is empty after converting to numeric and dropping NaNs")
    exit(1)
print(f"Train series length: {len(train_series)}, dtype: {train_series.dtype}")

AIC = []
SARIMAX_model = []
error_count = 0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_series,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit(disp=False)

            print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except Exception as e:
            # collect some error info for debugging but continue
            error_count += 1
            if error_count <= 5:
                print(f"Model fit error for {param} x {param_seasonal}: {e}")
            continue

if not AIC:
    print("Error: No valid SARIMAX models found. Check your data and parameters.")
    exit(1)

print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))
# Let's fit this model
best_model_idx = AIC.index(min(AIC))
# Use the cleaned numeric series for final fitting as well
mod = sm.tsa.statespace.SARIMAX(train_series,
                                order=SARIMAX_model[best_model_idx][0],
                                seasonal_order=SARIMAX_model[best_model_idx][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit(disp=False, maxiter=50, method='lbfgs')
results.plot_diagnostics(figsize=(20, 14))
plt.show()

# Helper to parse date strings (accepts dd/mm/YYYY or ISO formats)
def _parse_date(s):
    dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors='coerce')
    return dt

# Choose a start date for in-sample prediction and map to nearest index if needed
requested_start = _parse_date('31/12/2025')
if pd.isna(requested_start):
    requested_start = train_series.index[-1]

# Ensure start is within the sample index for get_prediction
if requested_start not in train_series.index:
    nearest_idx = train_series.index.get_indexer([requested_start], method='nearest')[0]
    start_loc = train_series.index[nearest_idx]
else:
    start_loc = requested_start

print(f"Using start location for get_prediction: {start_loc}")

pred0 = results.get_prediction(start=start_loc, dynamic=False)
pred0_ci = pred0.conf_int()
pred1 = results.get_prediction(start=start_loc, dynamic=True)
pred1_ci = pred1.conf_int()

# For forecasting beyond the sample, compute steps from last train index to requested end
requested_end = _parse_date('28/02/2026')
if pd.isna(requested_end):
    requested_end = train_series.index[-1]

# Estimate steps as number of days if index frequency is daily; fall back to 1 step minimum
last_idx = train_series.index[-1]
delta_days = (requested_end - last_idx).days
steps = max(1, delta_days) if (delta_days is not None) else 1
pred2 = results.get_forecast(steps=steps)
pred2_ci = pred2.conf_int()

# Ensure the forecast index is DatetimeIndex matching calendar dates
pred_mean = pred2.predicted_mean
if not isinstance(pred_mean.index, pd.DatetimeIndex):
    # infer frequency from train index, default to daily
    freq = train_series.index.inferred_freq
    if freq is None:
        freq = 'D'
    # start the forecast index the next period after last training index
    try:
        next_start = last_idx + pd.tseries.frequencies.to_offset(freq)
    except Exception:
        next_start = last_idx + pd.Timedelta(days=1)
    new_index = pd.date_range(start=next_start, periods=len(pred_mean), freq=freq)
    pred_mean.index = new_index
    # also set confidence interval index
    try:
        pred2_ci.index = new_index
    except Exception:
        pass

# Also ensure pred2.predicted_mean uses the new datetime index
try:
    pred2.predicted_mean.index = pred_mean.index
except Exception:
    try:
        pred2.predicted_mean = pred_mean
    except Exception:
        pass

# Ensure pred0 and pred1 predicted_mean indices are datetime-like for plotting/slicing
for p in (pred0, pred1):
    try:
        pm = p.predicted_mean
        if not isinstance(pm.index, pd.DatetimeIndex):
            # attempt to align with train_series index starting at start_loc
            freq = train_series.index.inferred_freq or 'D'
            try:
                start_idx = start_loc
            except NameError:
                start_idx = train_series.index[-1]
            new_idx = pd.date_range(start=start_idx, periods=len(pm), freq=freq)
            p.predicted_mean.index = new_idx
            try:
                p.conf_int().index = new_idx
            except Exception:
                pass
    except Exception:
        pass

try:
    print(pred_mean.loc[_parse_date('18/01/2026'):_parse_date('28/01/2026')])
except Exception:
    print(pred_mean)
# Build a forecast Series with a DatetimeIndex to allow date slicing and plotting
forecast_index = pred_mean.index if isinstance(pred_mean.index, pd.DatetimeIndex) else None
if forecast_index is None:
    freq = train_series.index.inferred_freq or 'D'
    try:
        next_start = last_idx + pd.tseries.frequencies.to_offset(freq)
    except Exception:
        next_start = last_idx + pd.Timedelta(days=1)
    forecast_index = pd.date_range(start=next_start, periods=len(pred_mean), freq=freq)
forecast_series = pd.Series(pred_mean.values, index=forecast_index)
try:
    pred2_ci.index = forecast_index
except Exception:
    pass

ax = data.plot(figsize=(20, 16))
pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
forecast_series.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(forecast_index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Monthly coffee price (USD)')
plt.xlabel('Date')
plt.legend()
plt.show()
plt.savefig('results/arima.png')

# Slice forecast by date range safely using the forecast_series (DatetimeIndex)
prediction = forecast_series.loc[_parse_date('18/01/2026'):_parse_date('28/01/2026')].values
# Build truth vector for the same date range as the prediction
start_dt = _parse_date('18/01/2026')
end_dt = _parse_date('28/01/2026')
truth_series = data['Lần cuối'].loc[start_dt:end_dt].astype(float).values

# Align lengths between truth and prediction
min_len = min(len(truth_series), len(prediction))
if min_len == 0:
    print('Warning: no overlapping truth values for the prediction period. Cannot compute MAE.')
    MAE = float('nan')
else:
    truth_aligned = truth_series[:min_len]
    pred_aligned = prediction[:min_len]
    # Avoid division by zero in MAPE
    nonzero_mask = truth_aligned != 0
    if nonzero_mask.sum() == 0:
        print('Warning: all truth values are zero in the selected period. MAE undefined.')
        MAE = float('nan')
    else:
        MAE = np.mean(np.abs(truth_aligned[nonzero_mask] - pred_aligned[nonzero_mask]))

print('The Mean Absolute Error for the forecast of year 2026 is {:.2f}'.format(MAE))