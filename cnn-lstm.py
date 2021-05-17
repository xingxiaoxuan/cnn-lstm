from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from get_data import input
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras .models import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional, Dense
from keras.utils.vis_utils import plot_model

origin_data = input()
# data = origin_data[:, [3, 4, 6]]
# print(data.shape)

# 取冷热电数据
# cool = data[:, 0]
# steam = data[:, 1]
# elec = data[:, 2]

# 加入噪声
normal_factor = 1
data = origin_data + normal_factor * np.random.normal(loc=0, scale=1, size=origin_data.shape)

MIN, MAX = {}, {}
for i in range(origin_data.shape[1]):
    MIN[i] = min(origin_data[:, i])
    MAX[i] = max(origin_data[:, i])
    data[:, i] = np.clip(data[:, i], MIN[i], MAX[i])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

data_train = scaled[: int(len(data) * 0.8)]
data_test = scaled[int(len(data) * 0.8):]
seq_len = 24
X_train = np.array([data_train[i: i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, 6] for i in range(data_train.shape[0] - seq_len)])
X_test = np.array([data_test[i: i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
y_test = np.array([data_test[i + seq_len, 6] for i in range(data_test.shape[0] - seq_len)])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (6984, 24, 13) (6984,) (1728, 24, 13) (1728,)

# time_step = 24
# train_noisy = elec_noisy[:8700]
# test_noisy = elec_noisy[8700-time_step:]
# train = elec[:8700]
# test = elec[8700-time_step:]

# def create_dataset(dataset, look_back):
#     dataX, dataY = [], []
#     for i in range(look_back, len(dataset)):
#         a = dataset[i-look_back:i]
#         dataX.append(a)
#     dataX = np.array(dataX)
#     dataX = dataX.reshape((dataX.shape[0], 1, dataX.shape[1]))
#     return np.array(dataX)
#
#
# train_noisy = create_dataset(train_noisy, 24)
# test_noisy = create_dataset(test_noisy, 24)
# train = create_dataset(train, 24)
# test = create_dataset(test, 24)

# x = []
# for n in range(len(elec) // 168):
#     a = np.zeros((7, 24))
#     for i in range(7):
#         for j in range(24):
#             a[i][j] = elec_noisy[168 * n + 24 * i + j]
#     x.append(a)

# x = np.array(x)
# print(x.shape)  # (52, 7, 24)

# print(train_noisy.shape, test_noisy.shape, train.shape, test.shape)
# (8700, 1, 24) (60, 1, 24) (8700, 1, 24) (60, 1, 24)


def build_model():
    # 模型参数设置
    TIME_STEPS = 24
    INPUT_DIM = 13
    lstm_units = 64
    batch_size = 256
    epochs = 100

    inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
    # drop1 = Dropout(0.3)(inputs)

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    # padding = 'same'
    # x = Conv1D(filters=128, kernel_size=5, activation='relu')(output1)#embedded_sequences
    x = MaxPooling1D(pool_size=3)(x)
    # x = Conv1D(filters=32, kernel_size=1, activation='relu')(inputs)
    # x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.2)(x)
    print(x.shape)

    lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # lstm_out = LSTM(lstm_units,activation='relu')(x)
    print(lstm_out.shape)

    output = Dense(1, activation='sigmoid')(lstm_out)
    # output = Dense(10, activation='sigmoid')(drop2)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(model, to_file='cude.png')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
    model.save('./cube')
    encoder = model(inputs=inputs, outputs=x)
    return model, encoder


model, encoder = build_model()
y_pred = model.predict(X_test)

# invert scaling for forecast
scaler.fit_transform(data[:, 6].reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.plot(y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()

# calculate RMSE、MAPE
RMSE = sqrt(mean_squared_error(y_test, y_pred))
MAPE = mean_absolute_error(y_test, y_pred)
print('Test RMSE: %.3f ，Test MAPE: %.3f' % (RMSE, MAPE))

acc = (1 - np.mean(np.abs((y_test - y_pred) / y_test))) * 100
print("精准度：", acc, "%")
