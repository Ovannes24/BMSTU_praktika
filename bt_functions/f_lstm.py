import pandas as pd
import numpy as np
import datetime as dt

from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, normalize

def LSTM_pred(df, train_size=150, epochs=5):
    df_train = df.iloc[:-train_size]
    df_test = df.iloc[-train_size:]
    
    data = df.Close.values
    training_data_len = df_train.shape[0]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)) 
    
    train_data = scaled_data[0:training_data_len]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i])
        y_train.append(train_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    
    model = Sequential()
    model.add(LSTM(50,return_sequences = True, input_shape = (x_train.shape[1],x_train.shape[2])))
    model.add(LSTM(50,return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(x_train, y_train, batch_size=1, epochs=epochs)
    
    test_data = scaled_data[training_data_len - 60:]

    
    x_test = []
    y_test = data[training_data_len:]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
    
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    return df_train, df_test, pd.DataFrame(predictions, index=df_test.index, columns=['Close'])


#End
