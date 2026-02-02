import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
try:
    from keras.layers import Dense, Activation, Dropout, LSTM
    from keras.models import Sequential
except Exception:
    from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
    from tensorflow.keras.models import Sequential

from common.model_evaluation import model_evaluation


class Conf:
    EPOCHS = 100
    SEQ_LEN = 50
    PREDICT_STEP = 20
    TRAIN_DATA_RATE = 0.9
    BATCH_SIZE = 500
    LAYERS = [1, 50, 100, 1]


def load_data(filename):
    df = pd.read_csv(filename)
    # Data trong file từ mới đến cũ, reverse để từ cũ đến mới
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Lưu dates để dùng cho plot
    dates = df['Ngày'].values
    
    # Chỉ lấy cột "KL" (Closing price alternative)
    # Convert string to float and handle suffixes like 'K', 'M', 'B'
    import re

    def parse_kl(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s == '':
            return np.nan
        # remove commas
        s = s.replace(',', '')
        mult = 1.0
        # handle suffixes
        if len(s) > 0 and s[-1].upper() in ('K', 'M', 'B'):
            suffix = s[-1].upper()
            s = s[:-1]
            if s == '':
                s = '0'
            if suffix == 'K':
                mult = 1e3
            elif suffix == 'M':
                mult = 1e6
            elif suffix == 'B':
                mult = 1e9
        # strip any remaining non-numeric chars
        s = re.sub(r'[^0-9eE\+\-\.]', '', s)
        if s == '' or s in ('-', '+'):
            return np.nan
        try:
            return float(s) * mult
        except Exception:
            return np.nan

    data = df['KL'].apply(parse_kl).astype(float).values.reshape(-1, 1)

    result = []
    base_prices = []  # Lưu giá gốc của mỗi window để denormalize

    # Tính số cửa sổ (windows) tổng cộng và vị trí chia train/test
    total_windows = len(data) - Conf.SEQ_LEN - 1
    row = round(total_windows * Conf.TRAIN_DATA_RATE)

    # Fit MinMaxScaler chỉ trên phần dữ liệu thô được sử dụng cho tập train
    # Phần dữ liệu thô cuối cùng được train sử dụng có chỉ số = row - 1 + SEQ_LEN
    train_raw_end_idx = row + Conf.SEQ_LEN
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_raw_end_idx])

    # Áp dụng transform cho toàn bộ dữ liệu sau khi fit trên train
    data_normalized = scaler.transform(data)

    for index in range(len(data_normalized) - Conf.SEQ_LEN - 1):
        window = data_normalized[index: index + Conf.SEQ_LEN + 1]
        # flatten window so resulting array has shape (window_len,) not (window_len, 1)
        result.append(window.flatten())
        base_prices.append(float(data[index][0]))  # Lưu giá gốc đầu tiên của window

    result = np.array(result)

    # Tách test base prices TRƯỚC khi shuffle (nếu có shuffle)
    test_base_prices = base_prices[int(row):]

    train = result[:int(row), :]

    _X_train = train[:, :-1]
    _y_train = train[:, -1]
    _X_test = result[int(row):, :-1]
    _y_test = result[int(row):, -1]

    # 增加一列
    _X_train = _X_train[:, :, np.newaxis]
    _X_test = _X_test[:, :, np.newaxis]

    # Lưu base prices tương ứng cho test set và dates tương ứng
    # test_dates bắt đầu từ vị trí Conf.SEQ_LEN + int(row) trong dates
    test_dates_start_idx = Conf.SEQ_LEN + int(row)
    test_dates = dates[test_dates_start_idx:test_dates_start_idx + len(_y_test)]
    # Reverse test_dates vì data đã được reverse từ mới->cũ sang cũ->mới
    test_dates = test_dates[::-1]

    print(_X_train.shape)
    print(_X_test.shape)
    return [_X_train, _y_train, _X_test, _y_test, test_base_prices, test_dates, scaler]


def normalise_windows(window_data):
    """
    Lưu ý: Hàm này không còn được sử dụng vì đã normalize trước khi tạo windows
    """
    pass


def denormalise_value(normalised_val, scaler):
    """
    Denormalize giá trị từ range [0, 1] về giá trị gốc
    """
    # Reshape thành 2D array (1, 1) để compatible với scaler
    val_2d = np.array([[normalised_val]])
    return scaler.inverse_transform(val_2d)[0, 0]


def build_model(layers):
    model = Sequential()

    model.add(LSTM(units=layers[1], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    predict = model.predict(data)
    print(predict.shape)
    predict = np.reshape(predict, (len(predict),))
    print(predict.shape)
    return predict


def predict_sequences_multiple(model, data, window_size, predict_len):
    prediction_seqs = []
    for i in range(int(len(data) / predict_len)):
        curr_frame = data[i * predict_len]
        predicted = []
        for j in range(predict_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def predict_sequence_full(model, data, window_size):
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_next_n_steps(model, data, window_size, n_steps):
    curr_frame = data[-1]  # 使用最后一个数据窗口
    predicted = []
    for i in range(n_steps):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def plot_results(y_true, y_pred, filename=None, future_pred=None, dates=None, full_prices=None, full_dates=None, test_start_idx=None):
    from matplotlib.ticker import StrMethodFormatter
    
    fig = plt.figure(facecolor='white', figsize=(16, 6))
    ax = fig.add_subplot(111)
    
    # Vẽ toàn bộ dữ liệu gốc làm nền
    if full_prices is not None and full_dates is not None:
        ax.plot(range(len(full_prices)), full_prices, label='Full Price Series', color='gray', alpha=0.4, linewidth=2)
    
    # Tính vị trí bắt đầu của test data trong full series
    if test_start_idx is None and full_prices is not None:
        # Nếu không biết test_start_idx, dùng vị trí sao cho test data ở cuối
        test_start_idx = len(full_prices) - len(y_true) - (len(future_pred) if future_pred is not None else 0)
    
    # Vẽ true data và prediction ở vị trí cuối của full series
    x_timeline = list(range(test_start_idx, test_start_idx + len(y_true))) if test_start_idx is not None else list(range(len(y_true)))
    
    ax.plot(x_timeline, y_true, label='True Data', linewidth=2)
    ax.plot(x_timeline, y_pred, label='Prediction', linewidth=2)
    
    # Nếu có dữ liệu dự báo tiếp theo, thêm vào plot
    if future_pred is not None:
        # Tính vị trí bắt đầu cho dự báo tiếp theo
        start_idx = x_timeline[-1]
        x_future = list(range(start_idx, start_idx + len(future_pred)))
        ax.plot(x_future, future_pred, label='Future Prediction', linestyle='--', marker='o', linewidth=2)
    
    # Format trục Y để hiển thị đơn vị $
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    # Thiết lập labels cho trục
    if dates is not None and len(dates) > 0:
        # Hiển thị một số dates quan trọng trên trục X
        step = max(1, len(dates) // 10)  # Hiển thị khoảng 10 labels
        tick_indices = list(range(0, len(dates), step))
        if len(dates) - 1 not in tick_indices:
            tick_indices.append(len(dates) - 1)
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([dates[i] if i < len(dates) else '' for i in tick_indices], rotation=45, ha='right')
        ax.set_xlabel(f'Timeline ({dates[0]} to {dates[-1]})', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('Timeline (Days)', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title('Coffee Price Forecast (14/01/2008 - 28/01/2026)', fontsize=14, fontweight='bold')
    
    # Thêm grid để dễ đọc
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


def plot_results_multiple(y_true, y_pred, predict_len, filename_prefix=None):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(y_true, label='True Data')
    for i, data in enumerate(y_pred):
        padding = [None for p in range(i * predict_len)]
        plt.plot(padding + data, label='Prediction')
    plt.legend()

    if filename_prefix:
        dirpath = os.path.dirname(filename_prefix) or '.'
        os.makedirs(dirpath, exist_ok=True)
        out = f"{filename_prefix}.png"
        plt.savefig(out)
        plt.close(fig)
        print(f"> Saved plot to {out}")
    else:
        plt.show()


def main():
    global_start_time = time.time()

    print('> Loading data... ')

    # sin: sin.csv; stock: stock.csv; coffee: coffee_price.csv
    filename = '/workspaces/copper_price_forecast/lstm/demo/coffee_price.csv'
    X_train, y_train, X_test, y_test, test_base_prices, dates, scaler = load_data(filename)

    print('> Data Loaded. Compiling...')

    model = build_model(Conf.LAYERS)

    model.fit(X_train, y_train, batch_size=Conf.BATCH_SIZE, epochs=Conf.EPOCHS, validation_split=0.05)

    # 预测一步
    predicted = predict_point_by_point(model, X_test)
    # 预测所有步
    # predicted = predict_sequence_full(model, X_test, Conf.SEQ_LEN)
    # 预测Conf.SEQ_LEN步
    # predicted = predict_sequences_multiple(model, X_test, Conf.SEQ_LEN, 50)

    print('Training duration (s) : ', time.time() - global_start_time)

    # Denormalize y_test và predicted về giá trị gốc sử dụng scaler
    # Reshape arrays thành (N, 1) để compatible với scaler
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predicted_denorm = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()

    # 预测接下来的 10 个值
    future_predicted = predict_next_n_steps(model, X_test, Conf.SEQ_LEN, 10)
    # Denormalize future predictions sử dụng scaler
    future_prices = scaler.inverse_transform(np.array(future_predicted).reshape(-1, 1)).flatten()
    
    print("\n=== Forecast next 10 days coffee prices ===")
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: ${price:.2f}")

    # 预测一步及所有步 - lưu đồ thị vào thư mục demo/results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    # Đọc lại toàn bộ giá và ngày để plot nền
    df_full = pd.read_csv(filename)
    df_full = df_full.iloc[::-1].reset_index(drop=True)
    # Sử dụng cùng parser để đọc full series (handle '0.00K' etc.)
    import re
    def _parse_series(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace(',', '')
        mult = 1.0
        if len(s) > 0 and s[-1].upper() in ('K', 'M', 'B'):
            suffix = s[-1].upper()
            s = s[:-1]
            if s == '':
                s = '0'
            if suffix == 'K': mult = 1e3
            if suffix == 'M': mult = 1e6
            if suffix == 'B': mult = 1e9
        s = re.sub(r'[^0-9eE\+\-\.]', '', s)
        try:
            return float(s) * mult
        except Exception:
            return np.nan

    full_prices = df_full['KL'].apply(_parse_series).astype(float).values
    full_dates = df_full['Ngày'].values

    plot_results(y_test_denorm, predicted_denorm, filename=os.path.join(results_dir, 'prediction.png'), future_pred=future_prices, dates=dates, full_prices=full_prices, full_dates=full_dates)
    # 预测Conf.SEQ_LEN步
    # plot_results_multiple(y_test, predicted, Conf.SEQ_LEN, filename_prefix=os.path.join(results_dir, 'prediction_multiple'))

    # 该模型评估方法不适合多步预测（适合所有步）
    model_evaluation(pd.DataFrame(y_test_denorm), pd.DataFrame(predicted_denorm))


if __name__ == '__main__':
    main()
