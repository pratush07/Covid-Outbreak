from datetime import datetime
import warnings
from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

plt.rcParams["figure.figsize"] = (25, 25)
plt.rcParams['font.size'] = '10'


# load dataset
def parser(x):
    return datetime.strptime(x, '%d-%m-%Y')


def read_file():
    df = read_csv("E:/Master's/Semester 1/Machine Learning/Project/Covid-Outbreak/Cases.csv", header=0, index_col=0,
                  parse_dates=True, squeeze=True,
                  date_parser=parser)
    return df


def clean_data(df):
    # clean cases data and convert them to numeric
    # df_cases = df.applymap(lambda x: x.strip())
    # df_cases = df_cases.applymap(lambda x: x.replace(",", ""))
    # df_cases = df_cases.applymap(lambda x: x.replace("..", ""))

    cols = df.columns[1:]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_cases = df.fillna(0)
    return df_cases


def arima_analysis(series, arima_order):
    counties = series.columns
    series.index = series.index.to_period('W')

    j, i = 1, 1

    fig, axs = plt.subplots(4, 7, constrained_layout=True)
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)
    # fig(figsize=(8, 6), dpi=80)

    for county in counties:
        X = series[county].values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()

        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))

        # evaluate forecasts
        MSE = mean_squared_error(test, predictions)
        RMSE = sqrt(MSE)
        #print(f'Test MSE & RMSE for county {county}:{MSE:.3f} , {RMSE :.3f}')
        print(f'For county {county} : MSE -> {MSE:.3f} RMSE-> {RMSE:.3f}')
        # plot forecasts against actual outcomes
        axs[i - 1, j - 1].set_title(f"{county}", fontsize=8)
        axs[i - 1, j - 1].plot(test, label="Expected Value")
        axs[i - 1, j - 1].plot(predictions, color='red', label="Predicted value")
        # axs[i-1,j-1].legend()
        axs[i - 1, j - 1].set_xlabel("Week #", fontsize=8)
        # selabel("")
        axs[i - 1, j - 1].set_ylabel("No. of Cases", fontsize=8)
        if (j == 7):
            i += 1
            j = 0
        j += 1

    # fig.legend(loc='upper center')

    fig.suptitle("Predictions and Actual Cases for various Counties")
    plt.show()


def hyper_testing(df_clean):
    #warnings.filterwarnings("ignore")
    p_values = range(0, 10)
    q_values = range(0, 3)
    d_values = range(0, 3)
    best_score, best_cfg = float("inf"), None
    X = df_clean.loc[:, 'Dublin']
    for p in p_values:
        for q in q_values:
            for d in d_values:
                order = (p, q, d)
                try:
                    train_size = int(len(X) * 0.66)
                    train, test = X[0:train_size], X[train_size:]
                    history = [x for x in train]
                    # make predictions
                    predictions = list()
                    for t in range(len(test)):
                        model = ARIMA(history, order=order)
                        model_fit = model.fit()
                        yhat = model_fit.forecast()[0]
                        predictions.append(yhat)
                        history.append(test[t])
                    # calculate out of sample error
                    RMSE = sqrt(mean_squared_error(test, predictions))
                    if RMSE < best_score:
                        best_score, best_cfg = RMSE, order
                        print('ARIMA%s RMSE=%.3f' % (order, RMSE))
                except:
                    continue

    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return order


def future_prediction(df):
    counties = df.columns
    predictions = {}

    for county in counties:
        X = df[county].values
        prediction = []
        #X = df['Cork'].values
        history = [x for x in X]
        for t in range(0, 4):
            model = ARIMA(history, order=(2, 2, 0))
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=4)[0]
            #predictions[f'{county}'].append(yhat)
            prediction.append(round(yhat))
            #print(f"Last value of Dataset : {history[-1]}")
            #print(f"Predicted value : {yhat:.0f}")
            history.append(round(yhat))
        predictions[county]= prediction
    predictions_sorted ={}
    predictions_sorted=sorted(predictions.items(), key=lambda x: x[1],reverse=True)
    print("The Counties predicted to have the Highest Cases in next four weeks are :")
    for i in range(3):
        print(f'{predictions_sorted[i][0]} -> {predictions_sorted[i][1]} ')




def main():
    df = read_file()
    df_clean = clean_data(df)
    order=hyper_testing(df_clean)
    # arima_analysis(df_clean, order)
    #arima_analysis(df_clean,(2,2,0))
    #future_prediction(df_clean)


if __name__ == "__main__":
    main()

