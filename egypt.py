import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def import_data():
    df = pd.read_csv('time_series_covid19_confirmed_global.csv')

    # egypt_data = df['Country/Region'] == 'Egypt'
    # print(df[egypt_data])

    egypt_data = df[df['Country/Region'] == 'Egypt'].T
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(egypt_data)


def plot_data():
    df = pd.read_csv('egypt_2.csv')

    x = df['number_of_days']
    y = df['total_infected']
    y1 = df['total_death']
    y2 = df['total_infected'] - (df['total_death'] + df['recovery'])

    fig, ax = plt.subplots(figsize=(10, 8))

    plot0 = ax.plot(x, y, label='Total infected')
    plot1 = ax.plot(x, y1, label='Total death')
    plot2 = ax.plot(x, y2, label='Active')

    ax.set_xticks(x,)
    ax.set_yticks([y_tick for y_tick in range(y.min(), y.max(), 20)])
    ax.set_xlabel('Days starting 1st-March')
    ax.set_ylabel('Total cases')
    ax.set_title('Total COVID-19 infections numbers in Egypt')

    ax.legend()
    plt.show()


def predict(day_number):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('egypt_2.csv')
    # print(df.head())
    lm = LinearRegression()
    x = df[['number_of_days']]
    # x = df['number_of_days']
    y = df['total_infected']
    errors = []
    coef = []
    # for degree in range(2, 8):
    poly = PolynomialFeatures(3)
    # poly.fit_transform(x)

    lm.fit(poly.fit_transform(x), y)
    y_hat = lm.predict(poly.fit_transform(x))
    # y_hat = lm.predict()
    # print(y_hat)
    r_sqr = lm.score(poly.fit_transform(x), y)
    mse = mean_squared_error(y, y_hat)
    # errors.append([r_sqr, mse])
    # print(r_sqr)
    coef.append([lm.intercept_, lm.coef_])
    #
    # ax1 = sns.distplot(y, hist=False, color='r', label='Actual')
    # sns.distplot(y_hat, hist=False, color='b', label='Fitted', ax=ax1)

    # plt.plot(x, y)
    # plt.plot(x, y_hat)

    plt.show()
    y_target = lm.intercept_
    for i in range(1, len(lm.coef_)):
        # print(lm.coef_[i], i)
        y_target += lm.coef_[i] * (day_number ** i)
    print('Expected:', y_target)
    # for error in errors:
    #     print(error)
    # for co in coef:
        # print(co)


def plot_bar():
    df = pd.read_csv('egypt.csv')
    # df.dropna(axis=0)
    x = df['number_of_days']
    labels = df['date']
    fig, ax = plt.subplots()
    width = 0.35
    rect1 = ax.bar(
        x - width/2, df['new_postive'],
        width, label='new_postive'
    )
    rect2 = ax.bar(
        x + width / 2, df['new_death'],
        width, label='new_death'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_data()
    # predict(42)
    plot_bar()
