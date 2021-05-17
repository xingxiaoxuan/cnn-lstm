from get_data import input
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout, Concatenate
from keras.models import Input, Model, load_model
from keras import layers, optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import tensorflow as tf

origin_data = input()
data = origin_data[0:2159, [4, 6]]


def get_data():
    # 加入噪声
    normal_factor = 1
    noisy_data = data + normal_factor * np.random.normal(loc=0, scale=1, size=data.shape)

    MIN, MAX = {}, {}
    for i in range(data.shape[1]):
        MIN[i] = min(data[:, i])
        MAX[i] = max(data[:, i])
        noisy_data[:, i] = np.clip(noisy_data[:, i], MIN[i], MAX[i])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(noisy_data)

    data_train = scaled[: int(len(data) * 0.8)]
    data_test = scaled[int(len(data) * 0.8):]
    return data_train, data_test


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


data_train, data_test = get_data()
# 热负荷数据
steam_train_noisy, _ = create_dataset(np.array(data_train[:, 0]), 24)
steam_test_noisy, _ = create_dataset(np.array(data_test[:, 0]), 24)
steam_train, steam_y_train = create_dataset(np.array(data_train[:, 0]), 24)
steam_test, steam_y_test = create_dataset(np.array(data_test[:, 0]), 24)
# print(steam_train_noisy.shape, steam_test_noisy.shape,
# steam_train.shape, steam_y_train.shape, steam_test.shape, steam_y_test.shape)
# (1703, 1, 24) (1703, 1, 24) (408, 1, 24) (408,)

# 电负荷数据
elec_train_noisy, _ = create_dataset(np.array(data_train[:, 1]), 24)
elec_test_noisy, _ = create_dataset(np.array(data_test[:, 1]), 24)
elec_train, elec_y_train = create_dataset(np.array(data_train[:, 1]), 24)
elec_test, elec_y_test = create_dataset(np.array(data_test[:, 1]), 24)
# print(elec_train_noisy.shape, elec_test_noisy.shape,
#       elec_train.shape, elec_y_train.shape, elec_test.shape, elec_y_test.shape)
# (1703, 1, 24) (1703, 1, 24) (408, 1, 24) (408,)
y_train = np.concatenate((steam_y_train.reshape(1703, 1), elec_y_train.reshape(1703, 1)), axis=1)
y_test = np.concatenate((steam_y_test.reshape(408, 1), elec_y_test.reshape(408, 1)), axis=1)
print(y_train.shape, y_test.shape)

# steam_train_noisy = tf.expand_dims(steam_train_noisy, axis=-1)
# steam_train = tf.expand_dims(steam_train, axis=-1)
# steam_test_noisy = tf.expand_dims(steam_test_noisy, axis=-1)
# steam_test = tf.expand_dims(steam_test, axis=-1)


def steam_DAE_model():
    input_tensor = Input(shape=(steam_train_noisy.shape[1], steam_train_noisy.shape[2]))
    hidden = LSTM(72, return_sequences=True, name='hidden_layer1')(input_tensor)
    # hidden = Dense(100, activation='softmax')(input_tensor)
    output_tensor = Dense(24, activation='sigmoid')(hidden)
    model = Model(input_tensor, output_tensor)
    model.summary()
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mae', metrics=['accuracy'])
    # fit network
    history = model.fit(steam_train_noisy, steam_train, epochs=200, batch_size=72,
                        validation_data=(steam_test_noisy, steam_test), shuffle=False)
    model.save('./steam_DAE')
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return model


def elec_DAE_model():
    input_tensor = Input(shape=(elec_train_noisy.shape[1], elec_train_noisy.shape[2]))
    hidden = LSTM(72, return_sequences=True, name='hidden_layer1')(input_tensor)
    # hidden_1 = LSTM(72, return_sequences=True)(input_tensor)
    # hidden_2 = Dropout(0.2)(hidden_1)
    # hidden = LSTM(48, return_sequences=True, name='hidden_layer2')(hidden_2)
    # hidden = Dense(100, activation='softmax')(input_tensor)
    output_tensor = Dense(24, activation='sigmoid')(hidden)
    model = Model(input_tensor, output_tensor)
    model.summary()
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mae', metrics=['accuracy'])
    # fit network
    history = model.fit(elec_train_noisy, elec_train, epochs=100, batch_size=72,
                        validation_data=(elec_test_noisy, elec_test), shuffle=False)
    model.save('./elec_DAE')
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return model


def build_auto_encoder():
    # steam_dae = steam_DAE_model()
    steam_extract = load_model('./steam_DAE')
    elec_extract = load_model('./elec_DAE')

    ext1 = steam_extract.get_layer('hidden_layer1').output
    ext2 = elec_extract.get_layer('hidden_layer2').output

    m = Concatenate(axis=1)([ext1, ext2])
    m = layers.Flatten()(m)
    predictions = Dense(2, activation='relu')(m)
    autoencoder = Model(inputs=[steam_extract.input, elec_extract.input], outputs=predictions)
    autoencoder.summary()
    for layer in steam_extract.layers:
        layer.trainable = False
    for layer in elec_extract.layers:
        layer.trainable = False

    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    autoencoder.compile(optimizer=adam, loss='mae', metrics=['accuracy'])
    history = autoencoder.fit([steam_train_noisy, elec_train_noisy], y_train, epochs=100, batch_size=72,
                              validation_data=([steam_test_noisy, elec_test_noisy], y_test), shuffle=False)
    autoencoder.save('./autoencoder')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return autoencoder


def single_predict():
    steam_extract = steam_DAE_model()
    # steam_extract = load_model('./steam_DAE')
    # elec_extract = load_model('./elec_DAE')

    ext1 = steam_extract.get_layer('hidden_layer1').output
    # ext2 = elec_extract.get_layer('hidden_layer2').output

    # m = Concatenate(axis=1)([ext1, ext2])
    m = layers.Flatten()(ext1)
    predictions = Dense(1, activation='relu')(m)
    autoencoder = Model(inputs=steam_extract.input, outputs=predictions)
    autoencoder.summary()
    for layer in steam_extract.layers:
        layer.trainable = False
    # for layer in elec_extract.layers:
    #     layer.trainable = False

    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    autoencoder.compile(optimizer=adam, loss='mae', metrics=['accuracy'])
    history = autoencoder.fit(steam_train_noisy, steam_y_train, epochs=100, batch_size=72,
                              validation_data=(steam_test_noisy, steam_y_test), shuffle=False)
    autoencoder.save('./single_steam_prediction')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def single_assess(xtest, ytest):
    model = load_model('./single_steam_prediction')
    y_pred = model.predict(xtest)

    # invert scaling for forecast
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data[:, 0].reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(ytest.reshape(-1, 1))

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


def muti_assess(test):
    # build_auto_encoder()
    Autoencoder = load_model('./autoencoder')
    y_pred = Autoencoder.predict([steam_test_noisy, elec_test_noisy])
    print(y_pred.shape)  # (408, 2)

    scaler = MinMaxScaler(feature_range=(0, 1))
    # invert scaling for forecast
    scaler.fit_transform(data)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(test)

    plt.plot(range(len(y_pred)), y_test[:, 0], label='steam_test', color='r')
    plt.plot(range(len(y_pred)), y_pred[:, 0], label='steam_prediction', color='y')
    plt.legend(['steam_test', 'steam_prediction'], loc='upper right')
    plt.figure()
    plt.plot(range(len(y_pred)), y_test[:, 1], label='elec_test', color='b')
    plt.plot(range(len(y_pred)), y_pred[:, 1], label='elec_prediction', color='g')
    plt.legend(['elec_test', 'elec_prediction'], loc='upper right')
    plt.legend()
    plt.show()

    # calculate RMSE、MAPE
    steam_RMSE = sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    steam_MAPE = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    print('Test RMSE: %.3f ，Test MAPE: %.3f' % (steam_RMSE, steam_MAPE))

    elec_RMSE = sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    elec_MAPE = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    print('Test RMSE: %.3f ，Test MAPE: %.3f' % (elec_RMSE, elec_MAPE))

    steam_acc = (1 - sqrt(np.mean(np.power(((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0]), 2)))) * 100
    print("精准度：", steam_acc, "%")
    elec_acc = (1 - np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1]))) * 100
    print("精准度：", elec_acc, "%")


# single_assess(steam_test_noisy, steam_y_test)
# steam_extract = steam_DAE_model()
