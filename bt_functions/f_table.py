import pandas as pd
import numpy as np
import datetime as dt
import talib

from sklearn.linear_model import LinearRegression

def redate_data(df, start, end, start_delta = 250, end_delta = 0):   
    """Редейт данных для исследования в определенном временном промежутке"""
    start_index = df.index.get_loc(df[start:end].iloc[0].name.strftime("%Y-%m-%d")) - start_delta
    end_index = df.index.get_loc(df[start:end].iloc[-1].name.strftime("%Y-%m-%d")) + 1 + end_delta
    return df.iloc[start_index:end_index]

def EMA_n(df, n):
    """Возвращает EMA с указанным периодом"""
    return pd.DataFrame(talib.EMA(df, timeperiod=n))

def RSI(df, n):
    """Возвращает RSI с указанным периодом"""
    gain = pd.Series(df).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return pd.DataFrame(100 - 100 / (1 + rs))

def cross(df1, df2):
    """Возвращает маску пеерсечения шортовой и лонговой торговли"""
    return (df1 != df1.shift(1)) & (df2 != df2.shift(1)) & (df2 != df1)

def RSI_short_arr(df):
    """Создает маску шортовой торговли для RSI стратегии"""
    isUP = False
    listUP = []

    for i in df:
        if i > 70:
            isUP = True
        if i < 30:
            isUP = False
        listUP.append(isUP)
    return pd.DataFrame(listUP, index=df.index)

def linreg(df):
    """Находит линейную регрессию для данных с названиями Close и Data_value (dv)"""
    lr = LinearRegression()
    X_train = np.array(df['dv']).reshape(-1, 1)
    y_train = df['Close']
    lr.fit(X_train, y_train)
    return lr.predict(X_train)

def regSplit(sdf, window1):
    regSplit = []
    c = 1
    TrueToFalse = False

    for i in sdf[f'RSI_min_{window1}']:
        if i <= 30:
            regSplit.append(c)
            TrueToFalse = True
        else:
            regSplit.append(np.nan)
            if TrueToFalse:
                c+=1
            TrueToFalse = False
    
    return regSplit

def profit(tdf, sdf):
    next_price = tdf.Close.shift(-1)
    next_price.iloc[-1] = sdf.Close.iloc[-1]

    past_price = tdf.Close

    profit = np.array((next_price/past_price)[tdf.long])
    
    return profit
