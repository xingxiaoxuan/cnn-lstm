from get_data import input
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Input, Model, load_model
from keras import layers, optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

origin_data = input()
data = origin_data[:, [3, 4, 6]]
print(data.shape)

# 归一化
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(data)

# 取冷热电数据
cool = data[:, 0]
steam = data[:, 1]
elec = data[:, 2]

MIN, MAX = min(elec), max(elec)

# 加入噪声
normal_factor = 1
elec_noisy = elec + normal_factor * np.random.normal(loc=0, scale=1, size=elec.shape)
elec_noisy = np.clip(elec_noisy, MIN, MAX)

scaler = MinMaxScaler(feature_range=(0, 1))
elec = scaler.fit_transform(elec.reshape(-1, 1))
elec_noisy = scaler.fit_transform(elec_noisy.reshape(-1, 1))

time_step = 24
train_noisy = elec_noisy[:8700]
test_noisy = elec_noisy[8700-time_step:]
train = elec[:8700]
test = elec[8700-time_step:]


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(look_back, len(dataset)):
        a = dataset[i-look_back:i]
        b = dataset[i]
        dataX.append(a)
        dataY.append(b)
    dataX, dataY = np.array(dataX), np.array(dataY)
    dataX = dataX.reshape((dataX.shape[0], 1, dataX.shape[1]))
    return np.array(dataX), dataY


train_noisy, _ = create_dataset(train_noisy, 24)
test_noisy, _ = create_dataset(test_noisy, 24)
train, y_train = create_dataset(train, 24)
test, y_test = create_dataset(test, 24)

print(train_noisy.shape, test_noisy.shape, train.shape, test.shape, y_train.shape, y_test.shape)
# (8700, 1, 24) (60, 1, 24) (8700, 1, 24) (60, 1, 24), (8676, 1) (60, 1)


def build_model():
    input_tensor = Input(shape=(train.shape[1], train.shape[2]))
    hidden = LSTM(72, return_sequences=True, name='hidden_layer')(input_tensor)
    # hidden = Dense(100, activation='softmax')(input_tensor)
    output_tensor = Dense(24, activation='sigmoid')(hidden)
    model = Model(input_tensor, output_tensor)
    model.summary()
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mae', metrics=['accuracy'])
    # fit network
    history = model.fit(train_noisy, train, epochs=100, batch_size=72,
                        validation_data=(test_noisy, test), shuffle=False)
    model.save('./DAE')
    model.save_weights('extract')
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    return model


# model = build_model()
model = load_model('./DAE')
# weight_Dense = model.get_layer('hidden_layer').get_weights()
# print(np.array(weight_Dense).shape)
# print(model.get_layer('hidden_layer').output)
# model.summary()
#
# for i, layer in enumerate(model.layers):
#     print(i, layer.name)
#
# yhat = model.predict(test_noisy)
#
# # invert scaling for forecast
# scaler.fit_transform(data[:, 2].reshape(-1, 1))
#
# # make a prediction
# total_mae = []
# for i in range(test.shape[0]):
#     inv_yhat = scaler.inverse_transform(yhat[i]).reshape(-1, 1)
#     inv_y = scaler.inverse_transform(test[i].reshape(-1, 1))
#     MAE = mean_absolute_error(inv_yhat, inv_y)
#     total_mae.append(MAE)
#     if i == 0:
#         # 做ROC曲线
#         plt.figure()
#         plt.plot(range(len(inv_yhat)), inv_yhat, 'b', label="predict")
#         plt.plot(range(len(inv_y)), inv_y, 'r', label="test")
#         plt.legend(['predict', 'test'], loc='upper right')  # 显示图中的标签
#         plt.show()
#
# print(np.mean(total_mae))


# encoder
def autoencoder():
    m = model.get_layer('hidden_layer').output
    # encoder = Model(inputs=inp, outputs=model.get_layer('hidden_layer').output)
    # m = encoder(inp)
    m = layers.Flatten()(m)
    predictions = Dense(1, activation='relu')(m)
    autoencoder = Model(inputs=model.input, outputs=predictions)
    autoencoder.summary()
    for layer in model.layers:
        layer.trainable = False

    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    autoencoder.compile(optimizer=adam, loss='mae', metrics=['accuracy'])
    history = autoencoder.fit(train_noisy, y_train, epochs=100, batch_size=72,
                              validation_data=(test_noisy, y_test), shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return autoencoder


Autoencoder = autoencoder()
y_pred = Autoencoder.predict(test_noisy)

# invert scaling for forecast
scaler.fit_transform(origin_data[:, 6].reshape(-1, 1))
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
