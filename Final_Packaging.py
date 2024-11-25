import pandas as pd
import numpy as np

# Load a CSV file into a DataFrame
df = pd.read_csv("/root/Capstone Signalling/datasets/Data/BTC_training_data_2023-06-01_to_2023-11-30.csv")
df = df.sort_values('date')
# Display the first few groups of the dataset
print(df.head())

# Add indicators
df['50_EMA'] = df['close'].ewm(span=50, adjust=False).mean()
df['200_EMA'] = df['close'].ewm(span=200, adjust=False).mean()
df['20_SMA'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()
df['upper_band'] = df['20_SMA'] + (df['stddev'] * 2)
df['lower_band'] = df['20_SMA'] - (df['stddev'] * 2)
delta = df['close'].diff(1)
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

# Define strategies
# Define buy/sell signal functions
def ema_bb_rsi_buy(group):
    return (group['RSI'] < 20) and (group['close'] <= group['lower_band']) and (group['50_EMA'] > group['200_EMA'])

def ema_bb_rsi_sell(group):
    return (group['RSI'] > 80) and (group['close'] >= group['upper_band']) and (group['50_EMA'] < group['200_EMA'])

def rsi_only_buy(group):
    return group['RSI'] < 30

def rsi_only_sell(group):
    return group['RSI'] > 70

# Add more functions for different combinations as needed

# Define strategies
strategies = [
    {"name": "EMA_BB_RSI", "buy_signal": ema_bb_rsi_buy, "sell_signal": ema_bb_rsi_sell},
    {"name": "RSI_Only", "buy_signal": rsi_only_buy, "sell_signal": rsi_only_sell},
    # Add more strategies here
]

# Define window periods
window_periods = [15, 60, 240, 480, 1440, 10080]  # 15 min, 1 hr, 4 hr, 8 hr, 1 day, 1 week

def ann_ret(df):
    # Annual Trade Return
    # Define starting and ending portfolio values
    starting_portfolio_value = df['portfolio_value'].iloc[0]
    ending_portfolio_value = df['portfolio_value'].iloc[-1]

    # Calculate total return over the period
    total_return = (ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value

    # Calculate the total period in minutes and convert to days
    total_minutes = (df.index[-1] - df.index[0]).total_seconds() / 60
    days_held = total_minutes / (24 * 60)  # Convert total minutes to days

    # Apply the annualized return formula based on the number of days
    annualized_return = (1 + total_return) ** (365 / days_held) - 1

    return(annualized_return)

def ann_vol(df):
    # Calculate minute-by-minute returns
    df['returns'] = df['portfolio_value'].pct_change()

    # Calculate the standard deviation of the returns
    std_dev_returns = df['returns'].std()

    # Annualize the volatility: multiply by the square root of the number of minutes in a year
    minutes_per_year = 525600  # 60 * 24 * 365
    annualized_volatility = std_dev_returns * np.sqrt(minutes_per_year)

    return(annualized_volatility)

def sr(ar, av):
    # Assuming the risk-free rate is 0 for simplicity; replace with an actual rate if available
    risk_free_rate = 0.043

    # Calculate the Sharpe Ratio
    sharpe_ratio = (ar - risk_free_rate) / av

    return(sharpe_ratio)
    

# Backtesting function
def backtest_strategy(df, buy_signal, sell_signal, window_period):
    index_count = 0
    df['chunk'] = np.arange(len(df)) // window_period + 1

    # Initialize parameters
    initial_balance = 10000
    balance = initial_balance
    btc_held = 0
    trade_returns = []

    for chunk, group in df.groupby('chunk'):
        buy_executed = False
        for i in range(len(group)):
            if buy_signal(group.iloc[i]) and not buy_executed and balance > 2000:
                # Buy logic
                amount_to_invest = balance - 2000
                btc_to_buy = (amount_to_invest / group['close'].iloc[i]) * (1 - 0.002)
                btc_held += btc_to_buy
                balance -= amount_to_invest
                buy_price = group['close'].iloc[i]
                buy_executed = True
                print(f"Buying BTC at {group['close'].iloc[i]:.2f} in chunk {chunk}, BTC held: {btc_held:.6f}, Balance: {balance:.2f}")

            elif (i >= 0 and i < (len(group)-1)) and sell_signal(group.iloc[i]) and btc_held > 0 and buy_executed:
                # Sell logic
                amount_to_receive = btc_held * group['close'].iloc[i] * (1 - 0.002)
                balance += amount_to_receive
                btc_held = 0
                trade_return = ((group['close'].iloc[i] - buy_price) / buy_price) * 100
                trade_returns.append(trade_return)
                print(f"Selling BTC at {group['close'].iloc[i]:.2f} in chunk {chunk}, Balance: {balance:.2f}, Trade Return: {trade_return:.2f}%")
                break
            elif (i == (len(group)-1)) and btc_held > 0 and buy_executed:
                # Sell logic
                amount_to_receive = btc_held * group['close'].iloc[i] * (1 - 0.002)
                balance += amount_to_receive
                btc_held = 0
                trade_return = ((group['close'].iloc[i] - buy_price) / buy_price) * 100
                trade_returns.append(trade_return)
                print(f"Selling BTC at {group['close'].iloc[i]:.2f} in chunk {chunk}, Balance: {balance:.2f}, Trade Return: {trade_return:.2f}%")
                break
            # Track balance, BTC held, and portfolio value at each step
            df.at[(group.index[i]), 'balance'] = balance
            df.at[(group.index[i]), 'btc_held'] = btc_held
            df.at[(group.index[i]), 'portfolio_value'] = balance + (btc_held * group['close'].iloc[i])  # Portfolio = cash + BTC value

    # Metrics calculation
    # portfolio_value = balance + btc_held * df['close'].iloc[-1]
    
    # Fill initial balance for rows where no action occurred
    df['balance'] = df['balance'].ffill()
    df['btc_held'] = df['btc_held'].ffill()

    # Ensure the date column is in datetime format and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    annualized_return = ann_ret(df)
    annualized_volatility = ann_vol(df)
    sharpe_ratio = sr(annualized_return, annualized_volatility)

    df.reset_index(inplace=True)

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio
    }

# Run strategies across all window periods
results = []
for strategy in strategies:
    for window_period in window_periods:
        metrics = backtest_strategy(
            df,
            strategy["buy_signal"],
            strategy["sell_signal"],
            window_period
        )
        results.append({
            "strategy": strategy["name"],
            "window_period": window_period,
            **metrics
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Find the best strategy
best_strategy = results_df.loc[results_df['sharpe_ratio'].idxmax()]

# Display results
print(results_df)
print("\nBest Strategy:")
print(best_strategy)