from get_data import input
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot
from keras.layers import LSTM, Dense, Dropout
from keras.models import Input, Model
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
elec += normal_factor * np.random.normal(loc=0, scale=1, size=elec.shape)
elec = np.clip(elec, MIN, MAX)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(elec.reshape(-1, 1))

time_step = 48
trainlist = scaled[:8700+time_step]
testlist = scaled[8700-time_step:]


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(look_back, len(dataset)):
        a = dataset[i-look_back:i]
        dataX.append(a)
        dataY.append(dataset[i])
    dataX, dataY = np.array(dataX), np.array(dataY)
    dataX = dataX.reshape((dataX.shape[0], 1, dataX.shape[1]))
    return np.array(dataX), np.array(dataY)


x_train, y_train = create_dataset(trainlist, time_step)
x_test, y_test = create_dataset(testlist, time_step)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (8700, 1, 48) (60, 1, 48) (8700,) (60,)


def build_model():
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(x_train_cool.shape[1], x_train_cool.shape[2]), return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(LSTM(128, return_sequences=False))
    # model.add(Dropout(0.3))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    input_tensor = Input(shape=(x_train.shape[1], x_train.shape[2]))
    # x = LSTM(128, return_sequences=True)(input_tensor)
    # x = Dropout(0.3)(x)
    # x = LSTM(64, return_sequences=False)(x)
    # x = Dropout(0.3)(x)
    x = LSTM(48, return_sequences=True)(input_tensor)
    x = Dropout(0.3)(x)
    x = LSTM(24, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(12, return_sequences=False)(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])
    # fit network
    history = model.fit(x_train, y_train, epochs=20, batch_size=72,
                        validation_data=(x_test, y_test), shuffle=False)
    # model.save('./steam_LSTM1')
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    return model


model = build_model()
# model = load_model('./steam-LSTM')

# make a prediction
yhat = model.predict(x_test)

# invert scaling for forecast
scaler.fit_transform(data[:, 2].reshape(-1, 1))
inv_yhat = scaler.inverse_transform(yhat)
inv_y = scaler.inverse_transform(y_test.reshape(-1, 1))

# 做ROC曲线
pyplot.figure()
pyplot.plot(range(len(inv_yhat)), inv_yhat, 'b', label="predict")
pyplot.plot(range(len(inv_yhat)), inv_y, 'r', label="test")
pyplot.legend(['predict', 'test'], loc='upper right')  # 显示图中的标签

# calculate RMSE、MAPE
RMSE = sqrt(mean_squared_error(inv_y, inv_yhat))
MAPE = mean_absolute_error(inv_y, inv_yhat)
print('Test RMSE: %.3f ，Test MAPE: %.3f' % (RMSE, MAPE))

acc = (1 - np.mean(np.abs((inv_y - inv_yhat) / inv_y))) * 100
print("精准度：", acc, "%")
pyplot.show()
