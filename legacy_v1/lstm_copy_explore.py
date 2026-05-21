import argparse
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
try:
    from keras.layers import Dense, Activation, Dropout, LSTM
    from keras.models import Sequential
except Exception:
    from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
    from tensorflow.keras.models import Sequential

from legacy_v1.src.model_evaluation import model_evaluation

cd /workspaces/copper_price_forecast && python3 -m lstm.demo.lstm_copy_explore --mode multi-step --use-volume --epochs 100 2>&1 | tail -50
class Conf:
    EPOCHS = 100
    SEQ_LEN = 50
    PREDICT_STEP = 10
    TRAIN_DATA_RATE = 0.9
    BATCH_SIZE = 128
    LAYERS = [1, 50, 100, 1]


def parse_volume(vol_str):
    """Parse volume string like '4.87K' to float (4870), handle empty strings"""
    if pd.isna(vol_str) or vol_str == '':
        return 0.0
    vol_str = str(vol_str).strip()
    if vol_str == '':
        return 0.0
    try:
        multiplier = 1.0
        if vol_str.endswith('K'):
            multiplier = 1000
            vol_str = vol_str[:-1]
        elif vol_str.endswith('M'):
            multiplier = 1_000_000
            vol_str = vol_str[:-1]
        return float(vol_str) * multiplier
    except:
        return 0.0


def load_data(filename, seq_len=Conf.SEQ_LEN, predict_step=Conf.PREDICT_STEP, scalers=None, fit_scaler_on_train=True, use_volume=False):
    df = pd.read_csv(filename)
    # Reverse từ mới->cũ thành cũ->mới
    df = df.iloc[::-1].reset_index(drop=True)
    dates = df['Ngày'].values
    
    # Close price
    close = df['Lần cuối'].astype(str).str.replace(',', '').astype(float).values
    
    # Volume (optional)
    if use_volume:
        volume = df['KL'].apply(parse_volume).values
        # Normalize volume to same scale: divide by max (avoid 0 division)
        vol_max = np.max(volume) if np.max(volume) > 0 else 1.0
        volume = volume / vol_max
        # Stack close + normalized volume
        data = np.column_stack([close, volume])  # (N, 2)
    else:
        data = close.reshape(-1, 1)  # (N, 1)

    # Tạo windows raw (trước khi scale)
    X_raw = []
    y_raw = []
    for i in range(len(data) - seq_len - predict_step + 1):
        X_raw.append(data[i: i + seq_len].copy())
        # Only use close (first column) for y
        y_raw.append(close[i + seq_len: i + seq_len + predict_step].reshape(-1, 1).copy())

    X_raw = np.array(X_raw)  # (N, seq_len, features)
    y_raw = np.concatenate(y_raw, axis=0)  # (N*predict_step, 1)
    y_raw = y_raw.reshape(-1, predict_step)  # (N, predict_step)

    total_windows = X_raw.shape[0]
    row = round(total_windows * Conf.TRAIN_DATA_RATE)

    # Create scalers if not provided
    if scalers is None:
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scalers = {'x': scaler_x, 'y': scaler_y}

    # Fit on train data if requested
    if fit_scaler_on_train:
        # fit X scaler on train X windows
        X_raw_for_fit = X_raw[:row].reshape(-1, X_raw.shape[2])  # (row*seq_len, features)
        scalers['x'].fit(X_raw_for_fit)
        
        # fit y scaler on train close values
        train_raw_end_idx = row + seq_len
        close_train = close[:train_raw_end_idx].reshape(-1, 1)
        scalers['y'].fit(close_train)

    # Transform X (all features)
    X_all_flat = X_raw.reshape(-1, X_raw.shape[2])
    X_all_scaled = scalers['x'].transform(X_all_flat).reshape(X_raw.shape)

    # Transform y (close prices only)
    y_all_flat = y_raw.reshape(-1, 1)
    y_all_scaled = scalers['y'].transform(y_all_flat).reshape(y_raw.shape)

    X_train = X_all_scaled[:row, :, :]
    y_train = y_all_scaled[:row, :]
    X_test = X_all_scaled[row:, :, :]
    y_test = y_all_scaled[row:, :]

    # test dates corresponding start index
    test_dates_start_idx = seq_len + row
    test_dates = dates[test_dates_start_idx:test_dates_start_idx + len(y_test)]
    test_dates = test_dates[::-1]

    # full prices and dates for plotting
    full_prices = close
    full_dates = dates

    return X_train, y_train, X_test, y_test, test_dates, scalers, full_prices, full_dates


def build_model_one_step(layers, n_features=1):
    model = Sequential()
    model.add(LSTM(units=layers[1], input_shape=(layers[1], n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=layers[3]))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def build_model_multi_step(seq_len, predict_steps, units1=50, units2=100, n_features=1):
    model = Sequential()
    model.add(LSTM(units=units1, input_shape=(seq_len, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units2, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=predict_steps))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def predict_recursive(model, X_test, window_size, n_steps):
    curr_frame = X_test[-1].copy()
    predicted = []
    for i in range(n_steps):
        p = model.predict(curr_frame[np.newaxis, :, :])[0, 0]
        predicted.append(p)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return np.array(predicted)


def plot_results(y_true, y_pred, filename=None, future_pred=None, dates=None, full_prices=None, full_dates=None, test_start_idx=None):
    fig = plt.figure(facecolor='white', figsize=(16, 6))
    ax = fig.add_subplot(111)

    if full_prices is not None and full_dates is not None:
        ax.plot(range(len(full_prices)), full_prices, label='Full Price Series', color='gray', alpha=0.4, linewidth=2)

    if test_start_idx is None and full_prices is not None:
        test_start_idx = len(full_prices) - len(y_true) - (len(future_pred) if future_pred is not None else 0)

    x_timeline = list(range(test_start_idx, test_start_idx + len(y_true))) if test_start_idx is not None else list(range(len(y_true)))

    ax.plot(x_timeline, y_true, label='True Data', linewidth=2)
    ax.plot(x_timeline, y_pred, label='Prediction', linewidth=2)

    if future_pred is not None:
        start_idx = x_timeline[-1]
        x_future = list(range(start_idx, start_idx + len(future_pred)))
        ax.plot(x_future, future_pred, label='Future Prediction', linestyle='--', marker='o', linewidth=2)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

    if dates is not None and len(dates) > 0:
        step = max(1, len(dates) // 10)
        tick_indices = list(range(0, len(dates), step))
        if len(dates) - 1 not in tick_indices:
            tick_indices.append(len(dates) - 1)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([dates[i] if i < len(dates) else '' for i in tick_indices], rotation=45, ha='right')
        ax.set_xlabel(f'Timeline ({dates[0]} to {dates[-1]})', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('Timeline (Days)', fontsize=12, fontweight='bold')

    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title('Coffee Price Forecast (Experiment)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    if filename:
        dirpath = os.path.dirname(filename) or '.'
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"> Saved plot to {filename}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['one-step', 'recursive', 'multi-step'], default='multi-step')
    parser.add_argument('--epochs', type=int, default=Conf.EPOCHS)
    parser.add_argument('--batch', type=int, default=Conf.BATCH_SIZE)
    parser.add_argument('--predict', type=int, default=Conf.PREDICT_STEP)
    parser.add_argument('--quick', action='store_true', help='Quick run with small data/epochs')
    parser.add_argument('--use-volume', action='store_true', help='Use volume feature from KL column')
    args = parser.parse_args()

    Conf.EPOCHS = args.epochs
    Conf.BATCH_SIZE = args.batch
    Conf.PREDICT_STEP = args.predict

    if args.quick:
        Conf.EPOCHS = 5
        Conf.BATCH_SIZE = 32

    filename = os.path.join(os.path.dirname(__file__), 'coffee_price.csv')

    print('> Loading data...')
    print(f'> Use volume: {args.use_volume}')
    X_train, y_train, X_test, y_test, dates, scalers, full_prices, full_dates = load_data(filename, seq_len=Conf.SEQ_LEN, predict_step=Conf.PREDICT_STEP, use_volume=args.use_volume)

    n_features = X_train.shape[2]
    scaler_y = scalers['y']
    print(f'Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}, n_features={n_features}')

    if args.mode == 'one-step' or args.mode == 'recursive':
        # adapt y shapes for one-step training (use first step)
        y_train_one = y_train[:, 0]
        y_test_one = y_test[:, 0]
        model = build_model_one_step([1, Conf.LAYERS[1], Conf.LAYERS[2], 1], n_features=n_features)
        model.fit(X_train, y_train_one, batch_size=Conf.BATCH_SIZE, epochs=Conf.EPOCHS, validation_split=0.05)

        pred = model.predict(X_test)
        pred = pred.reshape(-1)

        # denorm using y scaler
        y_test_denorm = scaler_y.inverse_transform(y_test_one.reshape(-1, 1)).flatten()
        pred_denorm = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()

        future_denorm = None
        if args.mode == 'recursive':
            future_norm = predict_recursive(model, X_test, Conf.SEQ_LEN, Conf.PREDICT_STEP)
            future_denorm = scaler_y.inverse_transform(future_norm.reshape(-1, 1)).flatten()

        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        mode_suffix = 'vol' if args.use_volume else 'base'
        plot_results(y_test_denorm, pred_denorm, filename=os.path.join(results_dir, f'prediction_{args.mode}_{mode_suffix}.png'), future_pred=future_denorm, dates=dates, full_prices=full_prices, full_dates=full_dates)
        model_evaluation(pd.DataFrame(y_test_denorm), pd.DataFrame(pred_denorm))

    elif args.mode == 'multi-step':
        model = build_model_multi_step(Conf.SEQ_LEN, Conf.PREDICT_STEP, units1=Conf.LAYERS[1], units2=Conf.LAYERS[2], n_features=n_features)
        model.fit(X_train, y_train, batch_size=Conf.BATCH_SIZE, epochs=Conf.EPOCHS, validation_split=0.05)

        pred = model.predict(X_test)
        # pred shape (N, predict_step)

        # Denormalize: flatten then inverse_transform then reshape
        pred_denorm = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
        y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        # choose first column (first-step) to compare series alignment for plotting
        y_test_first = y_test_denorm[:, 0]
        pred_first = pred_denorm[:, 0]

        # future prediction using last test input
        future_norm = model.predict(X_test[-1:])[0]
        future_denorm = scaler_y.inverse_transform(future_norm.reshape(-1, 1)).flatten()

        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        mode_suffix = 'vol' if args.use_volume else 'base'
        plot_results(y_test_first, pred_first, filename=os.path.join(results_dir, f'prediction_{args.mode}_{mode_suffix}.png'), future_pred=future_denorm, dates=dates, full_prices=full_prices, full_dates=full_dates)

        model_evaluation(pd.DataFrame(y_test_first), pd.DataFrame(pred_first))


if __name__ == '__main__':
    main()
