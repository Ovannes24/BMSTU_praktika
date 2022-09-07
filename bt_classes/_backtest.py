from bt_functions.f_table import redate_data, EMA_n, RSI, cross
from bt_functions.f_table import RSI_short_arr, linreg, regSplit, profit
from bt_functions.f_lstm import LSTM_pred

from bt_functions.f_plot import plot_arrows

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode, iplot

class BackTest:
    def __init__(self, df, start, end, start_delta = 250, end_delta = 0):
        self.df = df
        
        self.start= start
        self.end = end
        
        self.mdf = redate_data(self.df, start, end, start_delta=start_delta, end_delta=end_delta)
        self.mdf['dv'] = np.array(self.mdf.index.map(dt.datetime.toordinal))
        
        self.m_train, self.m_test, self.y_pred = LSTM_pred(self.mdf, train_size=300, epochs=1)
        
        self.mdf = pd.concat([self.m_train, self.y_pred])
        
        self.mdf['dv'] = np.array(self.mdf.index.map(dt.datetime.toordinal))
        
        
        
    def redate_mdf(self, start, end, start_delta=250, end_delta=0):
        self.mdf = redate_data(self.df, start, end, start_delta=start_delta, end_delta=end_delta)
    
    def make_strategy(self):
        self.sdf = self.mdf[['dv', 'Close']]
        
        self.sdf['RSI_14'] = RSI(self.sdf.Close, 14)
        self.sdf['RSI_14'].fillna(100)
        self.sdf['line_70'] = 70
        self.sdf['line_30'] = 30

        self.sdf['EMA_10'] = EMA_n(self.sdf.Close, 10)
        self.sdf['EMA_200'] = EMA_n(self.sdf.Close, 200)
        self.sdf['EMA_long'] = (self.sdf.EMA_10 >= self.sdf.EMA_200) & ~self.sdf.EMA_10.isna() & ~self.sdf.EMA_200.isna()
        self.sdf['EMA_short'] = (self.sdf.EMA_10 < self.sdf.EMA_200) & ~self.sdf.EMA_10.isna() & ~self.sdf.EMA_200.isna()

        window1=125
        self.sdf[f'RSI_max_{window1}'] = self.sdf.rolling(window1).RSI_14.max()
        self.sdf[f'RSI_min_{window1}'] = self.sdf.rolling(window1).RSI_14.min()

        self.sdf = self.sdf[self.sdf.notna().all(axis=1)]

        self.sdf['regSplit'] = regSplit(self.sdf, window1=window1)

        self.sdf['regSplitUp'] = np.nan
        self.sdf['regSplitDown'] = np.nan

        for i in np.sort(self.sdf['regSplit'].unique())[:-1]:
            mask = self.sdf['regSplit'] == i
            self.sdf.loc[mask, mask[mask].name] = pd.DataFrame(linreg(self.sdf.loc[mask]), index=mask[mask].index, columns=[mask[mask].name])

            delta = ((self.sdf.loc[mask, 'Close'] - self.sdf.loc[mask, mask[mask].name]).max() - (self.sdf.loc[mask, 'Close'] - self.sdf.loc[mask, mask[mask].name]).min())/2

            self.sdf.loc[mask,'regSplitUp'] = self.sdf.loc[mask, mask[mask].name] + delta
            self.sdf.loc[mask,'regSplitDown'] = self.sdf.loc[mask, mask[mask].name] - delta


        self.sdf.loc[self.y_pred.index, 'Close'] = self.m_test.Close

        self.sdf['RSI_short'] = RSI_short_arr(self.sdf['RSI_14'])
        self.sdf['RSI_long'] = ~self.sdf['RSI_short']

        self.sdf['sRSI'] = self.sdf[f'RSI_min_{window1}'] <= 30
        self.sdf['sEMA'] = ~self.sdf['sRSI']

        self.sdf['long'] = (self.sdf['RSI_long'] & self.sdf.sRSI) | (self.sdf.sEMA & self.sdf['EMA_long'])
        self.sdf['short'] = (self.sdf['RSI_short'] & self.sdf.sRSI) | (self.sdf.sEMA & self.sdf['EMA_short'])
        
        self.down = self.sdf[cross(self.sdf.long, self.sdf.short) & (self.sdf.short)]
        self.up =  self.sdf[cross(self.sdf.long, self.sdf.short) & (self.sdf.long)]
        
    def make_trade(self):
        self.tdf = self.sdf[cross(self.sdf.long, self.sdf.short)]
        self.tdf = self.tdf[['Close', 'long', 'short']]
        
        self.cash = 10_000

        self.profit_trades = profit(self.tdf, self.sdf)
        self.net_profit = (np.prod(self.profit_trades) - 1) * self.cash
        self.total_closed_trades = len(self.profit_trades)
        self.max_drawdown = np.max(self.profit_trades) * self.cash
        self.buy_and_hold_return = self.sdf.Close[self.start:self.end][-1]/self.sdf.Close[self.start:self.end][0] * self.cash
        self.profit_factor = (np.prod(self.profit_trades) - 1) / (1 - np.prod(self.profit_trades[self.profit_trades < 1]))
        
        
        self.name_PS = [
            'Net Profit [$]',
            'Net Profit [%]',
            'Total Closed Trades',
            'Max Drawdown [$]',
            'Buy & Hold Return [$]',
            'Buy & Hold Return [%]',
            'Profit Factor'
        ]
        
        self.PS = pd.Series(
        [
            self.net_profit,
            self.net_profit/self.cash * 100,
            self.total_closed_trades,
            self.max_drawdown,
            self.buy_and_hold_return - self.cash,
            (self.buy_and_hold_return - self.cash)/self.cash * 100,
            self.profit_factor
        ],
        name='PS', index=self.name_PS
    )
    
    
    def plot_bt(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.sdf.index, y=self.sdf.Close, name='Data'))

        plot_arrows(fig, self.up, 'up')
        plot_arrows(fig, self.down, 'down')

        fig.add_trace(go.Scatter(x=self.y_pred.index, y=self.y_pred.Close, line=dict(color='red'), name='prediction'))
        fig.add_trace(go.Scatter(x=self.m_test.index, y=self.m_test.Close, line=dict(color='green'), name='actual'))
        fig.add_trace(go.Scatter(x=self.sdf.index, y=self.sdf.regSplit, line=dict(color='purple'), name='chMID'))
        fig.add_trace(go.Scatter(x=self.sdf.index, y=self.sdf.regSplitUp, line=dict(color='purple'),name='chUP'))
        fig.add_trace(go.Scatter(x=self.sdf.index, y=self.sdf.regSplitDown, line=dict(color='purple'), name='chDOWN'))

        fig.add_trace(go.Scatter(x=self.sdf.index, y=self.sdf.EMA_10, line=dict(color='rgba(250,0,0, 0.2)'), name='EMA 10'))
        fig.add_trace(go.Scatter(x=self.sdf.index, y=self.sdf.EMA_200, line=dict(color='rgba(0,250,0, 0.2)'), name='EMA 200'))


        fig.update_layout(legend_orientation="h",
                          legend=dict(x=.5, y=1.05, xanchor="center"),
                          hovermode="x",
                          margin=dict(l=0, r=0, t=0, b=0),
                          height=600)

        fig.update_traces(hoverinfo="all", hovertemplate="X: %{x}<br>Y: %{y}")
        fig.show()