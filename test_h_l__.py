import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import datetime as dt
from datetime import date
import pandas_datareader.data as web
from matplotlib import style
from datetime import datetime,timedelta

def date_prep():
    N=90
    end=datetime.now()
    start=datetime.now()-timedelta(days=N)
    return start, end

def prepare_csv(comp_name):
    start, end = date_prep()
    try:
        df = web.DataReader(comp_name, 'yahoo', start, end)
        df.to_csv('{}.csv'.format(comp_name), index=False)
    except Exception as e:
        print(e)

def find_AdjClose(df60_scaled,last_day,sc):
    i="AdjClose"
    arr=[]
    arr.append(last_day[0])
    m = keras.models.load_model('StockPredictorSBIN.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0,0])
    m = keras.models.load_model('StockPredictorTCS.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorWIPRO.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorUNITEDBNK.BO{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorRELIANCE.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^BSESN{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^NSEI{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    return arr
def find_Close(df60_scaled,last_day,sc):
    i = "Close"
    arr = []
    arr.append(last_day[0])
    m = keras.models.load_model('StockPredictorSBIN.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorTCS.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorWIPRO.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorUNITEDBNK.BO{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorRELIANCE.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^BSESN{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^NSEI{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    return arr
def find_Open(df60_scaled,last_day,sc):
    i = "Open"
    arr = []
    arr.append(last_day[0])
    m = keras.models.load_model('StockPredictorSBIN.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorTCS.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorWIPRO.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorUNITEDBNK.BO{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorRELIANCE.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^BSESN{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^NSEI{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    return arr
def find_Low(df60_scaled, last_day,sc):
    i = "Low"
    arr = []
    arr.append(last_day[0])
    m = keras.models.load_model('StockPredictorSBIN.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorTCS.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorWIPRO.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorUNITEDBNK.BO{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorRELIANCE.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^BSESN{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^NSEI{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    return arr
def find_High(df60_scaled, last_day,sc):
    i = "High"
    arr = []
    arr.append(last_day[0])
    m = keras.models.load_model('StockPredictorSBIN.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorTCS.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorWIPRO.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorUNITEDBNK.BO{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictorRELIANCE.NS{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^BSESN{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    m = keras.models.load_model('StockPredictor^NSEI{}.model'.format(i))
    pred = m.predict(df60_scaled)
    pred = sc.inverse_transform(pred)
    arr.append(pred[0, 0])
    return arr

def predict_stock(comp_name):
    #comp_name = input("Enter the company name ")
    comp_name = comp_name.upper()
    prepare_csv(comp_name)
    df = pd.read_csv('{}.csv'.format(comp_name))
    df60 = df.iloc[-60:, ].values

    sc = MinMaxScaler()
    sc.fit(df60)
    df60_scaled= sc.fit_transform(df60)
    df60_scaled = np.array(df60_scaled)
    df60_scaled = np.reshape(df60_scaled, (-1, df60_scaled.shape[0], 6))

    last_day= df.iloc[-1:, 5].values #Adj close last day
    output = df.iloc[-60:, 5].values #Adj Close
    sc1 = MinMaxScaler()
    output = np.reshape(output, (-1, 1))
    sc1.fit(output)
    sc1.fit_transform(output)
    adjcloses=[]
    adjcloses=find_AdjClose(df60_scaled,last_day,sc1)
    last_day=[]
    output=[]

    last_day= df.iloc[-1:, 3].values #Close last day
    output = df.iloc[-60:, 3].values #Close
    sc2 = MinMaxScaler()
    output = np.reshape(output, (-1, 1))
    sc2.fit(output)
    sc2.fit_transform(output)
    closes=[]
    closes=find_Close(df60_scaled,last_day,sc2)
    last_day=[]
    output=[]

    last_day= df.iloc[-1:, 2].values #Open last day
    output = df.iloc[-60:, 2].values #Open
    sc3 = MinMaxScaler()
    output = np.reshape(output, (-1, 1))
    sc3.fit(output)
    sc3.fit_transform(output)
    opens=[]
    opens=find_Open(df60_scaled,last_day,sc3)
    last_day=[]
    output=[]

    last_day= df.iloc[-1:, 1].values #Low last day
    output = df.iloc[-60:, 1].values #Low
    sc4 = MinMaxScaler()
    output = np.reshape(output, (-1, 1))
    sc4.fit(output)
    sc4.fit_transform(output)
    lows=[]
    lows=find_Low(df60_scaled,last_day,sc4)
    last_day=[]
    output=[]

    last_day= df.iloc[-1:, 0].values #High last day
    output = df.iloc[-60:, 0].values #High
    sc5 = MinMaxScaler()
    output = np.reshape(output, (-1, 1))
    sc5.fit(output)
    sc5.fit_transform(output)
    highs=[]
    highs=find_High(df60_scaled,last_day,sc5)
    last_day=[]
    output=[]

    print("ADj Close :")
    print(adjcloses)
    print("Close :")
    print(closes)
    print("Open :")
    print(opens)
    print("High :")
    print(highs)
    print("Low :")
    print(lows)

    os.remove('{}.csv'.format(comp_name))

predict_stock("SBIN.NS")