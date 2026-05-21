import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import sys
try:
    from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
    from tensorflow.keras.models import Sequential
    _BACKEND = 'tensorflow'
except Exception:
    try:
        from keras.layers import Dense, Activation, Dropout, LSTM
        from keras.models import Sequential
        _BACKEND = 'keras'
    except Exception:
        sys.exit("Required package 'tensorflow' or 'keras' not found. Install with: pip install tensorflow (recommended) or pip install keras")

from legacy_v1.src.model_evaluation import model_evaluation


class Conf:
    EPOCHS = 500
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
    
    # Chỉ lấy cột "Lần cuối" (Closing price)
    # Convert string to float (remove comma separator)
    data = df['Lần cuối'].astype(str).str.replace(',', '').astype(float).values.reshape(-1, 1)

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
        result.append(window)
        base_prices.append(float(data[index][0]))  # Lưu giá gốc đầu tiên của window

    result = np.array(result)

    # Tách test base prices TRƯỚC khi shuffle (nếu có shuffle)
    test_base_prices = base_prices[int(row):]

    train = result[:int(row), :]

    _X_train = train[:, :-1]
    _y_train = train[:, -1]
    _X_test = result[int(row):, :-1]
    _y_test = result[int(row):, -1]

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
    curr_frame = data[-1] 
    predicted = []
    for i in range(n_steps):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def plot_results(y_true, y_pred, filename=None, future_pred=None, dates=None, full_prices=None, full_dates=None, test_start_idx=None):
    # seaborn optional (fallback nếu không cài)
    try:
        import seaborn as sns
    except Exception:
        sns = None
    import matplotlib.dates as mdates
    from matplotlib.dates import YearLocator, DateFormatter

    # Áp dụng style ggplot
    plt.style.use('ggplot')
    if sns is not None:
        try:
            sns.set_palette("husl")
        except Exception:
            pass

    fig = plt.figure(facecolor='white', figsize=(16, 8))
    ax = fig.add_subplot(111)

    # Chuyển full_dates sang datetime (thử dayfirst rồi fallback)
    full_dates_dt = None
    if full_dates is not None:
        full_dates_dt = pd.to_datetime(full_dates, dayfirst=True, errors='coerce')

    # Tính vị trí bắt đầu của test data trong full series
    # note: do not subtract future_pred here, we want the true data aligned to the latest dates
    if test_start_idx is None and full_prices is not None:
        test_start_idx = len(full_prices) - len(y_true)

    # Vẽ toàn bộ dữ liệu gốc làm nền (dùng datetime nếu có)
    if full_prices is not None and full_dates_dt is not None:
        mask = ~full_dates_dt.isna()
        ax.plot(full_dates_dt[mask], full_prices[mask], label='Full Price Series',
                color='#E63946', alpha=0.7, linewidth=1.8)
    elif full_prices is not None:
        ax.plot(range(len(full_prices)), full_prices, label='Full Price Series',
                color='#E63946', alpha=0.7, linewidth=1.8)

    # Vẽ true data và prediction tại vị trí cuối của full series (dùng datetime nếu có)
    if full_dates_dt is not None:
        x_timeline_dates = full_dates_dt[test_start_idx: test_start_idx + len(y_true)]
        ax.plot(x_timeline_dates, y_true, label='Actual Price', linewidth=2.5, marker='o',
                markersize=3, color='#2E86AB', alpha=0.9)
        ax.plot(x_timeline_dates, y_pred, label='Prediction', linewidth=2.5, marker='s',
                markersize=3, color='#A23B72', alpha=0.7)
    else:
        x_timeline = list(range(test_start_idx, test_start_idx + len(y_true))) if test_start_idx is not None else list(range(len(y_true)))
        ax.plot(x_timeline, y_true, label='Actual Price', linewidth=2.5, marker='o',
                markersize=3, color='#2E86AB', alpha=0.9)
        ax.plot(x_timeline, y_pred, label='Prediction', linewidth=2.5, marker='s',
                markersize=3, color='#A23B72', alpha=0.7)

    # Nếu có dữ liệu dự báo tiếp theo, thêm vào plot (tạo ngày tiếp theo nếu dùng datetime)
    if future_pred is not None:
        if full_dates_dt is not None:
            # Try to create future dates starting the day after last_date; fallback to numeric indices if invalid
            try:
                last_date = full_dates_dt[test_start_idx + len(y_true) - 1]
                if pd.isna(last_date):
                    raise ValueError('last_date is NaT')
                # add a buffer so forecast doesn't visually overlap the last points
                gap_days = 1  # could be unrelated; ensures clear separation
                start_future = last_date + pd.Timedelta(days=gap_days + 1)
                x_future = pd.date_range(start=start_future, periods=len(future_pred), freq='D')
                ax.plot(x_future, future_pred, label='Future Forecast', linestyle='--',
                        marker='D', markersize=4, linewidth=2.5, color='#F18F01', alpha=0.85)
                ax.fill_between(x_future, future_pred, alpha=0.15, color='#F18F01')
            except Exception:
                # fallback to numeric indices placed after the last plotted index
                # also add gap of len(y_true) to separate clearly
                start_idx = test_start_idx + len(y_true) + len(y_true)
                x_future = list(range(start_idx, start_idx + len(future_pred)))
                ax.plot(x_future, future_pred, label='Future Forecast', linestyle='--',
                        marker='D', markersize=4, linewidth=2.5, color='#F18F01', alpha=0.85)
                ax.fill_between(x_future, future_pred, alpha=0.15, color='#F18F01')
        else:
            # Start future points after the last plotted index to avoid overlap
            start_idx = x_timeline[-1] + 1
            x_future = list(range(start_idx, start_idx + len(future_pred)))
            ax.plot(x_future, future_pred, label='Future Forecast', linestyle='--',
                marker='D', markersize=4, linewidth=2.5, color='#F18F01', alpha=0.85)
            ax.fill_between(x_future, future_pred, alpha=0.15, color='#F18F01')

    # Format trục Y để hiển thị đơn vị $
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

    # Thiết lập trục X hiển thị theo năm
    if full_dates_dt is not None:
        ax.xaxis.set_major_locator(YearLocator())            # mỗi năm một mốc chính
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))    # chỉ hiển thị năm
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,7)))  # minor ticks (tuỳ chọn)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('Timeline (Days)', fontsize=12, fontweight='bold')

    ax.set_ylabel('Daily coffee price (USD)', fontsize=12, fontweight='bold')
    
    # Cải thiện grid theo style ggplot
    ax.grid(True, alpha=0.35, linestyle='-', linewidth=0.6, color='white')
    ax.set_axisbelow(True)
    ax.set_facecolor('#E8E8E8')

    # Cải thiện legend
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='none')

    plt.tight_layout()

    if filename:
        dirpath = os.path.dirname(filename) or '.'
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
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

def plot_train_test_split(filename):
    """
    Vẽ biểu đồ phân chia dữ liệu Train/Test dựa trên tỷ lệ TRAIN_DATA_RATE
    """
    # Đọc dữ liệu
    df = pd.read_csv(filename)
    # Đảo ngược dữ liệu như trong load_data
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Xử lý dữ liệu giá
    prices = df['Lần cuối'].astype(str).str.replace(',', '').astype(float).values
    dates = df['Ngày'].values
    
    # Tính toán điểm cắt (Split Point)
    # Lưu ý: Cần trừ đi SEQ_LEN vì load_data cắt bớt đoạn đầu để tạo window
    data_len = len(prices) - Conf.SEQ_LEN - 1
    train_size = round(data_len * Conf.TRAIN_DATA_RATE)
    split_idx = train_size + Conf.SEQ_LEN  # Điểm bắt đầu của test set trên trục thời gian thực
    
    # Vẽ biểu đồ
    plt.figure(figsize=(14, 6))
    
    # Vẽ phần Train
    plt.plot(range(split_idx), prices[:split_idx], label='Training Set', color='blue', linewidth=2)
    
    # Vẽ phần Test
    plt.plot(range(split_idx, len(prices)), prices[split_idx:], label='Test Set', color='orange', linewidth=2)
    
    # Đánh dấu điểm cắt
    plt.axvline(x=split_idx, color='red', linestyle='--', label='Split Point')
    
    # Trang trí biểu đồ
    plt.title('Data Split: Training Set vs Test Set', fontsize=14, fontweight='bold')
    plt.xlabel('Timeline (Days)', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lưu hoặc hiển thị
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'data_split.png')
    plt.savefig(save_path)
    print(f"> Saved train/test split plot to {save_path}")
    plt.show() # Hoặc plt.close() nếu không muốn hiện cửa sổ

def correlation(x, y):
    """
    Tính hệ số tương quan Pearson thủ công (Manual implementation)
    """
    # === PHẦN SỬA ĐỔI ===
    # Đảm bảo input là list 1 chiều, tránh lỗi nếu đưa vào numpy array (N, 1)
    if hasattr(x, 'flatten'): x = x.flatten().tolist()
    if hasattr(y, 'flatten'): y = y.flatten().tolist()
    # ====================

    # Finding the mean of the series x and y
    mean_x = sum(x) / float(len(x))
    mean_y = sum(y) / float(len(y))
    
    # Subtracting mean from the individual elements
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    
    # covariance for x and y
    numerator = sum([sub_x[i] * sub_y[i] for i in range(len(sub_x))])
    
    # Standard Deviation of x and y
    std_deviation_x = sum([sub_x[i]**2.0 for i in range(len(sub_x))])
    std_deviation_y = sum([sub_y[i]**2.0 for i in range(len(sub_y))])
    
    # squaring by 0.5 to find the square root
    # Tránh lỗi chia cho 0 nếu độ lệch chuẩn bằng 0
    denominator = (std_deviation_x * std_deviation_y)**0.5
    if denominator == 0:
        return 0
        
    cor = numerator / denominator
    return cor

def main():
    global_start_time = time.time()

    print('> Loading data... ')

    # sin: sin.csv; stock: stock.csv; coffee: coffee_price.csv
    filename = '/workspaces/demo/coffee_price.csv'
    X_train, y_train, X_test, y_test, test_base_prices, dates, scaler = load_data(filename)
    plot_train_test_split(filename)
    print('> Data Loaded. Compiling...')

    model = build_model(Conf.LAYERS)
    model.fit(X_train, y_train, batch_size=Conf.BATCH_SIZE, epochs=Conf.EPOCHS, validation_split=0.05)

    predicted = predict_point_by_point(model, X_test)
    
    # predicted = predict_sequences_multiple(model, X_test, Conf.SEQ_LEN, 50)

    print('Training duration (s) : ', time.time() - global_start_time)

    # Denormalize y_test và predicted về giá trị gốc sử dụng scaler
    # Reshape arrays thành (N, 1) để compatible với scaler
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predicted_denorm = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()

    
    future_predicted = predict_next_n_steps(model, X_test, Conf.SEQ_LEN, 10)
    # Denormalize future predictions sử dụng scaler
    future_prices = scaler.inverse_transform(np.array(future_predicted).reshape(-1, 1)).flatten()
    
    print("\n=== Forecast next 10 days coffee prices ===")
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: ${price:.2f}")

    # lưu đồ thị vào thư mục demo/results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    # Đọc lại toàn bộ giá và ngày để plot nền
    df_full = pd.read_csv(filename)
    df_full = df_full.iloc[::-1].reset_index(drop=True)
    full_prices = df_full['Lần cuối'].astype(str).str.replace(',', '').astype(float).values
    full_dates = df_full['Ngày'].values

    plot_results(y_test_denorm, predicted_denorm, filename=os.path.join(results_dir, 'prediction_price.png'), future_pred=future_prices, dates=dates, full_prices=full_prices, full_dates=full_dates)
    # plot_results_multiple(y_test, predicted, Conf.SEQ_LEN, filename_prefix=os.path.join(results_dir, 'prediction_multiple'))
    r_score = correlation(y_test_denorm, predicted_denorm)
    print(f"\n> Correlation Coefficient (Manual Calculation): {r_score:.4f}")
    model_evaluation(pd.DataFrame(y_test_denorm), pd.DataFrame(predicted_denorm))

if __name__ == '__main__':
    main()