from get_data import input
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

origin_data = input()
data = origin_data[:, [3, 4, 6]]
print(data.shape)

# 归一化
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(data)

train, test = data[:8712, 2], data[-48:, 2]
history = [x for x in train]
predictions = list()


# history = [x for x in train[:, i]]
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    # obs = return_back(test[t])
    history.append(obs)
    print('predicted = %f, expected = %f' % (yhat, obs))

MSE = mean_squared_error(test, predictions)
MAE = mean_absolute_error(test, predictions)
RMSE = np.sqrt(mean_squared_error(test, predictions))  # RMSE就是对MSE开方即可

print('Test MSE: %.3f' % MSE)  # 15.533
print('Test MAE: %.3f' % MAE)  # 2.321
print('Test RMSE: %.3f' % RMSE)  # 3.941

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
