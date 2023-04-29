import yfinance as yf
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import concurrent.futures
from datetime import datetime, timedelta
from sklearn.model_selection import ParameterGrid
from concurrent.futures import ProcessPoolExecutor


# Function to download stock data
def download_data(ticker, start_date, end_date, interval='1h'):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    return stock_data

# Function to plot the stock price and buy/sell signals
def plot_signals(signals, ticker):
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(111, ylabel='Price in $')
    signals[['price', 'moving_avg', 'upper_band', 'lower_band']].plot(ax=ax1)
    ax1.plot(signals.loc[signals.positions == 1.0].index, signals.price[signals.positions == 1.0], '^', markersize=10, color='g', label='Buy')
    ax1.plot(signals.loc[signals.positions == -1.0].index, signals.price[signals.positions == -1.0], 'v', markersize=10, color='r', label='Sell')
    plt.title(f'{ticker} Bollinger Bands Mean Reversion Strategy')
    plt.legend(loc='best')
    plt.show()

# Function to implement the Bollinger Bands mean reversion strategy
def bollinger_bands_mean_reversion(stock_data, window, num_std_dev):
    signals = pd.DataFrame(index=stock_data.index)
    signals['price'] = stock_data['Adj Close']
    signals['moving_avg'] = signals['price'].rolling(window=window).mean()
    signals['std_dev'] = signals['price'].rolling(window=window).std()
    signals['upper_band'] = signals['moving_avg'] + (signals['std_dev'] * num_std_dev)
    signals['lower_band'] = signals['moving_avg'] - (signals['std_dev'] * num_std_dev)
    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['price'] < signals['lower_band'], 1.0, 0.0)
    signals['signal'] = np.where(signals['price'] > signals['upper_band'], -1.0, signals['signal'])
    signals['positions'] = signals['signal'].diff()
    return signals

def calculate_pnl(signals, initial_investment, transaction_fee_per_share):
    signals['position_diff'] = signals['positions'].diff()
    signals['fees'] = signals['position_diff'].abs() * transaction_fee_per_share
    signals['holdings'] = signals['positions'] * signals['price']
    signals['cash'] = initial_investment - (signals['position_diff'] * signals['price'] + signals['fees']).cumsum()
    signals['total'] = signals['cash'] + signals['holdings']

    signals['trade_duration'] = signals.positions[signals.positions != 0].groupby((signals.positions != 0).cumsum()).cumcount()
    signals['positions'] = np.where(signals.trade_duration >= 5, 0, signals.positions)

    return signals

def single_backtest(stock_data, window, num_std_dev, initial_investment, transaction_fee_per_share):
    if stock_data.empty:
        return 0, 0, 0

    signals = bollinger_bands_mean_reversion(stock_data, window, num_std_dev)
    signals = calculate_pnl(signals, initial_investment, transaction_fee_per_share)

    pnl = signals['total'][-1] - initial_investment
    profit_rate = pnl / initial_investment
    num_transactions = signals['positions'].abs().sum()

    return pnl, profit_rate, num_transactions

def backtest_strategy(ticker, start_date, end_date, window, num_std_dev, initial_investment, transaction_fee_per_share, n=100):
    pnl_list = []
    profit_rate_list = []
    num_transactions_list = []

    start_date = pd.to_datetime(start_date).replace(tzinfo=pytz.UTC)
    end_date = pd.to_datetime(end_date).replace(tzinfo=pytz.UTC)
    stock_data = download_data(ticker, start_date - timedelta(days=n), end_date)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(single_backtest, stock_data[start_date-timedelta(days=i):end_date-timedelta(days=i)], window, num_std_dev, initial_investment, transaction_fee_per_share) for i in range(n)]

        for future in concurrent.futures.as_completed(futures):
            pnl, profit_rate, num_transactions = future.result()
            pnl_list.append(pnl)
            profit_rate_list.append(profit_rate)
            num_transactions_list.append(num_transactions)

    mean_pnl = np.mean(pnl_list)
    mean_profit_rate = np.mean(profit_rate_list)
    mean_num_transactions = np.mean(num_transactions_list)

    return mean_pnl, mean_profit_rate, mean_num_transactions

def tune_parameters(ticker, start_date, end_date, initial_investment, transaction_fee_per_share, n=100):
    param_grid = {
        'window': list(range(10, 51, 5)),
        'num_std_dev': np.arange(1.5, 2.6, 0.1)
    }

    best_profit_rate = -np.inf
    best_params = None

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    stock_data = download_data(ticker, start_date - timedelta(days=n), end_date)

    with ProcessPoolExecutor() as executor:
        futures = []
        for params in ParameterGrid(param_grid):
            future = executor.submit(backtest_strategy, stock_data, start_date, end_date, params['window'], params['num_std_dev'], initial_investment, transaction_fee_per_share, n=n)
            futures.append((future, params))
        
        for future, params in futures:
            mean_pnl, mean_profit_rate, mean_num_transactions = future.result()
            if mean_profit_rate > best_profit_rate:
                best_profit_rate = mean_profit_rate
                best_params = params

    return best_params, best_profit_rate

# Implement the Bollinger Bands mean reversion strategy with the new parameters
def implement_strategy_with_new_params(ticker, start_date, end_date, best_params, initial_investment, transaction_fee_per_share):
    window = best_params['window']
    num_std_dev = best_params['num_std_dev']

    stock_data = download_data(ticker, start_date, end_date)
    signals = bollinger_bands_mean_reversion(stock_data, window, num_std_dev)
    signals = calculate_pnl(signals, initial_investment, transaction_fee_per_share)

    return signals
