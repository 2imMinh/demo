#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/6/8 20:01
@Description: 采用LSTM进行sin函数、股票（标准普尔500股权指数）及期铜预测
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
try:
    from keras.layers import Dense, Activation, Dropout, LSTM
    from keras.models import Sequential
except Exception:
    from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
    from tensorflow.keras.models import Sequential

from common.model_evaluation import model_evaluation


class Conf:
    # epochs
    EPOCHS = 500
    # 时间序列长度
    SEQ_LEN = 50
    # 预测步数
    PREDICT_STEP = 20
    # 测试训练集比例
    TRAIN_DATA_RATE = 0.9
    # 批大小
    BATCH_SIZE = 500
    # 网络形状
    LAYERS = [1, 50, 100, 1]


def load_data(filename):
    """
    数据准备
    """
    data = pd.read_csv(filename).values

    result = []
    for index in range(len(data) - Conf.SEQ_LEN - 1):
        result.append(data[index: index + Conf.SEQ_LEN + 1])
    # 数据标准化
    result = normalise_windows(result)

    result = np.array(result)

    row = round(result.shape[0] * Conf.TRAIN_DATA_RATE)
    train = result[:int(row), :]
    np.random.shuffle(train)

    _X_train = train[:, :-1]
    _y_train = train[:, -1]
    _X_test = result[int(row):, :-1]
    _y_test = result[int(row):, -1]

    # 增加一列
    _X_train = _X_train[:, :, np.newaxis]
    _X_test = _X_test[:, :, np.newaxis]

    print(_X_train.shape)
    print(_X_test.shape)
    return [_X_train, _y_train, _X_test, _y_test]


def normalise_windows(window_data):
    """
    对原始数据做标准化：n_i = (p_i/p0 - 1)
    对应的反标准化公式为：p_i = p_0 * (n_i + 1)
    兼容数据为 shape (N,) 或 (N, 1) 的情形。
    """
    normalised_data = []
    for window in window_data:
        # 保证为 1D float 数组（处理像 [ [v], [v], ... ] 的情况）
        arr = np.array(window, dtype=float).flatten()
        base = float(arr[0]) if arr.size > 0 else 1.0
        # 避免除以 0
        if base == 0.0:
            base = 1e-8
        normalised_window = [(float(p) / base) - 1.0 for p in arr]
        normalised_data.append(normalised_window)
    return normalised_data


def build_model(layers):
    """
    模型定义
    """
    model = Sequential()

    model.add(LSTM(units=layers[1], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=layers[3]))
    model.add(Activation("tanh"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    """
    每次预测1步
    """
    predict = model.predict(data)
    print(predict.shape)
    predict = np.reshape(predict, (len(predict),))
    print(predict.shape)
    return predict


def predict_sequences_multiple(model, data, window_size, predict_len):
    """
    每次预测Conf.SEQ_LEN步
    """
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
    """
    每次预测所有步
    """
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def plot_results(y_true, y_pred, filename=None):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(y_true, label='True Data')
    plt.plot(y_pred, label='Prediction')
    plt.legend()
    if filename:
        dirpath = os.path.dirname(filename) or '.'
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(filename)
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

    # sin: sin.csv; stock: stock.csv; copper: co_lstm.csv
    X_train, y_train, X_test, y_test = load_data('/workspaces/copper_price_forecast/lstm/demo/co_lstm.csv')

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

    # 预测一步及所有步 - lưu đồ thị vào thư mục demo/results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_results(y_test, predicted, filename=os.path.join(results_dir, 'prediction.png'))
    # 预测Conf.SEQ_LEN步
    # plot_results_multiple(y_test, predicted, Conf.SEQ_LEN, filename_prefix=os.path.join(results_dir, 'prediction_multiple'))

    # 该模型评估方法不适合多步预测（适合所有步）
    model_evaluation(pd.DataFrame(y_test), pd.DataFrame(predicted))


if __name__ == '__main__':
    main()
