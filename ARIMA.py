from datetime import datetime
from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


# load dataset
def parser(x):
    return datetime.strptime(x, '%d-%m-%Y')


series = read_csv("E:/Master's/Semester 1/Machine Learning/Project/Covid-Outbreak/Cases.csv", header=0, index_col=0, parse_dates=True, squeeze=True,
                  date_parser=parser)

counties = series.columns

series.index = series.index.to_period('W')
for county in counties:
    print(series[county])
    #series.index = pd.to_datetime(series.index)
    # split into train and test sets
    X = series[county].values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    #X.astype(str).astype(int)



    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
    plt.title(f"Predicition of Cases for county {county}")
    plt.plot(test, label="Expected Value")
    plt.plot(predictions, color='red', label="Predicted value")
    plt.legend()
    plt.xlabel("Week #")
    plt.ylabel("No. of Cases")
    plt.show()
