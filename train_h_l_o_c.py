# importing all important packages
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import datetime as dt
from datetime import date
import pandas_datareader.data as web
import os


def date_prep():
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    d = d1.split('/')
    day = int(d[0])
    month = int(d[1])
    year = int(d[2])
    year_19 = year - 19
    end = dt.datetime(year, month, day)
    if (month == 4 and day == 31) or (month == 6 and day == 31) or (month == 9 and day == 31) or (
            month == 11 and day == 31):
        start = dt.datetime(year_19, month, day - 1)
    else:
        if month == 2 and day == 29 and not (((year_19 % 4 == 0) and (year_19 % 100 != 0)) or (year_19 % 400 == 0)):
            start = dt.datetime(year_19, month, day - 1)
        else:
            start = dt.datetime(year_19, month, day)
    return start, end


def prepare_csv(comp_name):
    start, end = date_prep()
    try:
        df = web.DataReader(comp_name, 'yahoo', start, end)
        df.to_csv('{}.csv'.format(comp_name), index=False)
    except Exception as e:
        print(e)


def create_model(x_train):
    ####MODEL cretion #####
    model = Sequential()  # define the Keras model
    model.add(
        LSTM(units=240, return_sequences=True, input_shape=(x_train.shape[1], 6)))  # 120 neurons in the hidden layer
    ##return_sequences=True makes LSTM layer to return the full history including outputs at all times
    model.add(Dropout(0.2))
    model.add(LSTM(units=520, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=520, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    # adding optimizer and loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(comp_name):
    # reading a csv file
    prepare_csv(comp_name)
    df = pd.read_csv('{}.csv'.format(comp_name))

    df.dropna(how='any', inplace=True)

    training = df

    sc = MinMaxScaler()
    sc.fit(training)
    training_set_scaled = sc.fit_transform(training)

    X_train = []
    y1_train = []
    timestamp = 60
    length = len(training)

    #Open   2, High  0, Low   1, Close   3, Adj Close 5
    ################ adj close
    '''for i in range(timestamp, length + 1):
        X_train.append(training_set_scaled[i - timestamp:i, ])

    for i in range(timestamp, length + 1):
        y1_train.extend(training_set_scaled[i:i + 1, 5])
    x_train = []
    y1_train = np.array(y1_train)
    x_train = X_train[:y1_train.shape[0]]
    y1_train = np.reshape(y1_train, (-1, 1))
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    model1 = create_model(x_train)
    model1.fit(x_train, y1_train, epochs=5, batch_size=32)  # ,callbacks=[tensorboard])
    # storing the model
    model1.save('StockPredictor{}AdjClose.model'.format(comp_name))
    x_train = []
    y1_train = []'''

    #################close
    '''for i in range(timestamp, length + 1):
        X_train.append(training_set_scaled[i - timestamp:i, ])

    for i in range(timestamp, length + 1):
        y1_train.extend(training_set_scaled[i:i + 1, 3])
    x_train = []
    y1_train = np.array(y1_train)
    x_train = X_train[:y1_train.shape[0]]
    y1_train = np.reshape(y1_train, (-1, 1))
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    model1 = create_model(x_train)
    model1.fit(x_train, y1_train, epochs=5, batch_size=32)  # ,callbacks=[tensorboard])
    # storing the model
    model1.save('StockPredictor{}Close.model'.format(comp_name))
    x_train = []
    y1_train = []'''

    ############################LOW
    '''for i in range(timestamp, length + 1):
        X_train.append(training_set_scaled[i - timestamp:i, ])

    for i in range(timestamp, length + 1):
        y1_train.extend(training_set_scaled[i:i + 1, 1])
    x_train = []
    y1_train = np.array(y1_train)
    x_train = X_train[:y1_train.shape[0]]
    y1_train = np.reshape(y1_train, (-1, 1))
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    model1 = create_model(x_train)
    model1.fit(x_train, y1_train, epochs=5, batch_size=32)  # ,callbacks=[tensorboard])
    # storing the model
    model1.save('StockPredictor{}Low.model'.format(comp_name))
    x_train = []
    y1_train = []'''
    ##############High
    for i in range(timestamp, length + 1):
        X_train.append(training_set_scaled[i - timestamp:i, ])

    for i in range(timestamp, length + 1):
        y1_train.extend(training_set_scaled[i:i + 1, 0])
    x_train = []
    y1_train = np.array(y1_train)
    x_train = X_train[:y1_train.shape[0]]
    y1_train = np.reshape(y1_train, (-1, 1))
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    model1 = create_model(x_train)
    model1.fit(x_train, y1_train, epochs=5, batch_size=32)  # ,callbacks=[tensorboard])
    # storing the model
    model1.save('StockPredictor{}High.model'.format(comp_name))
    x_train = []
    y1_train = []
    ################## Open
    '''for i in range(timestamp, length + 1):
        X_train.append(training_set_scaled[i - timestamp:i, ])

    for i in range(timestamp, length + 1):
        y1_train.extend(training_set_scaled[i:i + 1, 2])
    x_train = []
    y1_train = np.array(y1_train)
    x_train = X_train[:y1_train.shape[0]]
    y1_train = np.reshape(y1_train, (-1, 1))
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    model1 = create_model(x_train)
    model1.fit(x_train, y1_train, epochs=5, batch_size=32)  # ,callbacks=[tensorboard])
    # storing the model
    model1.save('StockPredictor{}Open.model'.format(comp_name))
    x_train = []
    y1_train = []'''

    os.remove(comp_name + '.csv')


train_model('SBIN.NS')
train_model('TCS.NS')
train_model('WIPRO.NS')
train_model('UNITEDBNK.BO')
train_model('RELIANCE.NS')
train_model('^BSESN')
train_model('^NSEI')