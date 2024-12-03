import pandas as pd
import numpy as np

# Define strategies
# Define buy/sell signal functions
# There are separate buy and sell signals for each strategy separately which makes it easy to read and go to the strategy 
#       and also understand what's happenning inside. The indicator defnitions for each group will taken from the df 
#           which was defined at the start

def sma_only_buy(group):
    return((group['5_SMA'] > group['20_SMA']))

def sma_only_sell(group):
    return((group['5_SMA'] < group['20_SMA']))

def emacross520_buy(group):
    return (group['5_EMA'] > group['20_EMA'])

def emacross520_sell(group):
    return (group['5_EMA'] < group['20_EMA'])

def sma20_bb20_rsi2575_buy(group):
    return (group['RSI'] < 25) and (group['close'] <= group['lower_band']) and (group['close'] > group['20_SMA'])

def sma20_bb20_rsi2575_sell(group):
    return (group['RSI'] > 75) and (group['close'] >= group['upper_band']) and (group['close'] < group['20_SMA'])

def ema2050_bb50_rsi1585_buy(group):
    return (group['RSI'] < 15) and (group['close'] <= group['lower_band50']) and (group['20_EMA'] > group['50_EMA'])

def ema2050_bb50_rsi1585_sell(group):
    return (group['RSI'] > 85) and (group['close'] >= group['upper_band50']) and (group['20_EMA'] < group['50_EMA'])

def sma200_bb50_rsi2080_buy(group):
    return (group['RSI'] < 20) and (group['close'] <= group['lower_band50']) and (group['close'] > group['200_SMA'])

def sma200_bb50_rsi2080_sell(group):
    return (group['RSI'] > 80) and (group['close'] >= group['upper_band50']) and (group['close'] < group['200_SMA'])

def sma100_bb50_rsi2080_buy(group):
    return (group['RSI'] < 20) and (group['close'] <= group['lower_band50']) and (group['close'] > group['100_SMA'])

def sma100_bb50_rsi2080_sell(group):
    return (group['RSI'] > 80) and (group['close'] >= group['upper_band50']) and (group['close'] < group['100_SMA'])

def sma300_bb50_rsi2080_buy(group):
    return (group['RSI'] < 20) and (group['close'] <= group['lower_band50']) and (group['close'] > group['300_SMA'])

def sma300_bb50_rsi2080_sell(group):
    return (group['RSI'] > 80) and (group['close'] >= group['upper_band50']) and (group['close'] < group['300_SMA'])

def sma250_bb50_rsi3070_buy(group):
    return (group['RSI'] < 30) and (group['close'] <= group['lower_band50']) and (group['close'] > group['250_SMA'])

def sma250_bb50_rsi3070_sell(group):
    return (group['RSI'] > 70) and (group['close'] >= group['upper_band50']) and (group['close'] < group['250_SMA'])

######### Buy and Sell signals for the extra Strategies #############
# def sma200_bb20_rsi2080_buy(group):
#     return (group['RSI'] < 20) and (group['close'] <= group['lower_band']) and (group['close'] > group['200_SMA'])

# def sma200_bb20_rsi2080_sell(group):
#     return (group['RSI'] > 80) and (group['close'] >= group['upper_band']) and (group['close'] < group['200_SMA'])

# def sma200_bb50_rsi2575_buy(group):
#     return (group['RSI'] < 25) and (group['close'] <= group['lower_band50']) and (group['close'] > group['200_SMA'])

# def sma200_bb50_rsi2575_sell(group):
#     return (group['RSI'] > 75) and (group['close'] >= group['upper_band50']) and (group['close'] < group['200_SMA'])

# def ema1050_bb20_rsi1090_buy(group):
#     return (group['RSI'] < 10) and (group['close'] <= group['lower_band']) and (group['10_EMA'] > group['50_EMA'])

# def ema1050_bb20_rsi1090_sell(group):
#     return (group['RSI'] > 90) and (group['close'] >= group['upper_band']) and (group['10_EMA'] < group['50_EMA'])

# def sma100_bb_rsi_buy(group):
#     return((group['RSI'] < 30) & (group['close'] <= group['lower_band']) & (group['close'] > group['100_SMA']))

# def sma100_bb_rsi_sell(group):
#     return((group['RSI'] > 70) & (group['close'] >= group['upper_band']) & (group['close'] < group['100_SMA']))

# def emacross_buy(group):
#     return (group['20_EMA'] > group['50_EMA'])

# def emacross_sell(group):
#     return (group['20_EMA'] < group['50_EMA'])

# def ema_bb_rsi_buy(group):
#     return (group['RSI'] < 30) and (group['close'] <= group['lower_band']) and (group['50_EMA'] > group['200_EMA'])

# def ema_bb_rsi_sell(group):
#     return (group['RSI'] > 70) and (group['close'] >= group['upper_band']) and (group['50_EMA'] < group['200_EMA'])

# def ema20100_bb_rsi_buy(group):
#     return (group['RSI'] < 20) and (group['close'] <= group['lower_band50']) and (group['20_EMA'] > group['100_EMA'])

# def ema20100_bb_rsi_sell(group):
#     return (group['RSI'] > 80) and (group['close'] >= group['upper_band50']) and (group['20_EMA'] < group['100_EMA'])

# def rsi_only_buy(group):
#     return group['RSI'] < 30

# def rsi_only_sell(group):
#     return group['RSI'] > 70

# def sma_bb_rsi_buy(group):
#     return ((group['RSI'] < 30) &  (group['close'] <= group['lower_band']) & (group['20_SMA'] > group['50_SMA']))

# def sma_bb_rsi_sell(group):
#     return ((group['RSI'] > 70) &  (group['close'] >= group['lower_band']) & (group['20_SMA'] < group['50_SMA']))

# def sma100200_bb_rsi_buy(group):
#     #Generating Buy Signal Based on EMA, Bollinger Bands and RSI
#     return ((group['RSI'] < 30) &  (group['close'] <= group['lower_band']) & (group['100_SMA'] > group['200_SMA']))

# def sma100200_bb_rsi_sell(group):
#     #Generating Buy Signal Based on EMA, Bollinger Bands and RSI
#     return ((group['RSI'] > 70) &  (group['close'] >= group['lower_band']) & (group['100_SMA'] < group['200_SMA']))

# Define strategies
# Whenever the buy signal is called in the main trading loop from the strategy array, for each buy and sell, a function is triggered
#   which returns back the buy and sell signal values for each group(chunk) that is running inside the loop
strategies = [
    
    {"name": "SMACross_Only(5,20)", "buy_signal": sma_only_buy, "sell_signal": sma_only_sell},
    {"name": "EMACross_Only(5,20)", "buy_signal": emacross520_buy, "sell_signal": emacross520_sell},
    {"name": "SMA(20)_BB(20)_RSI(25,75)", "buy_signal": sma20_bb20_rsi2575_buy, "sell_signal": sma20_bb20_rsi2575_buy},
    {"name": "EMA(20,50)_BB(50)_RSI(15,85)", "buy_signal": ema2050_bb50_rsi1585_buy, "sell_signal": ema2050_bb50_rsi1585_sell},
    {"name": "SMA(200)_BB(50)_RSI(20,80)", "buy_signal": sma200_bb50_rsi2080_buy, "sell_signal": sma200_bb50_rsi2080_sell},
    {"name": "SMA(100)_BB(50)_RSI(20,80)", "buy_signal": sma100_bb50_rsi2080_buy, "sell_signal": sma100_bb50_rsi2080_sell},
    {"name": "SMA(300)_BB(50)_RSI(20,80)", "buy_signal": sma300_bb50_rsi2080_buy, "sell_signal": sma300_bb50_rsi2080_sell},
    {"name": "SMA(250)_BB(50)_RSI(30,70)", "buy_signal": sma250_bb50_rsi3070_buy, "sell_signal": sma250_bb50_rsi3070_sell}

    # Extra Strategies
    # {"name": "EMA(50,200)_BB(20)_RSI(30,70)", "buy_signal": ema_bb_rsi_buy, "sell_signal": ema_bb_rsi_sell},
    # {"name": "EMA(20,100)_BB(50)_RSI(20,80)", "buy_signal": ema20100_bb_rsi_buy, "sell_signal": ema20100_bb_rsi_sell},
    # {"name": "SMA(100,200)_BB(20)_RSI(30,70)", "buy_signal": sma100200_bb_rsi_buy, "sell_signal": sma100200_bb_rsi_sell},
    # {"name": "SMA(100)_BB(20)_RSI(30,70)", "buy_signal": sma100_bb_rsi_buy, "sell_signal": sma100_bb_rsi_sell},
    # {"name": "RSI_Only(30/70)", "buy_signal": rsi_only_buy, "sell_signal": rsi_only_sell},
    # {"name": "EMACross_Only(20,50)", "buy_signal": emacross_buy, "sell_signal": emacross_sell},
    # {"name": "EMA(10,50)_BB(20)_RSI(10,90)", "buy_signal": ema1050_bb20_rsi1090_buy, "sell_signal": ema1050_bb20_rsi1090_sell},
    # {"name": "SMA(200)_BB(20)_RSI(20,80)", "buy_signal": sma200_bb20_rsi2080_buy, "sell_signal": sma200_bb20_rsi2080_sell},
    # {"name": "SMA(200)_BB(50)_RSI(25,75)", "buy_signal": sma200_bb50_rsi2575_buy, "sell_signal": sma200_bb50_rsi2575_sell},
]

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
    # risk_free_rate = 0

    # Calculate the Sharpe Ratio
    sharpe_ratio = ar / av

    return(sharpe_ratio)

def pr(df, annualized_return):
    # 1. Calculate Buy-and-Hold Strategy Returns

    # Assume initial investment is equal to the strategy's initial balance
    initial_balance = 10000
    initial_price = df['close'].iloc[0]
    final_price = df['close'].iloc[-1]

    # Calculate the amount of BTC bought initially and then held until the end
    btc_bought = (initial_balance / initial_price) * (1 - 0.002)
    final_balance_buy_hold = btc_bought * final_price * (1 - 0.002)  # Applying sell transaction cost

    # Calculate Buy-and-Hold Return
    buy_hold_return = (final_balance_buy_hold - initial_balance) / initial_balance

    # 2. Calculate the Ratio of Strategy Return to Buy-and-Hold Return
    if buy_hold_return != 0:
        performance_ratio = annualized_return / buy_hold_return
    else:
        performance_ratio = float('inf')  # Handle case where buy-hold return is zero

    return(performance_ratio)
    

# Backtesting function
def backtest_strategy(df, buy_signal, sell_signal, window_period):
    index_count = 0
    df['chunk'] = np.arange(len(df)) // window_period + 1       # Divide the dataframe into chunks of the window size
    # Initialize parameters
    initial_balance = 10000         # Initial Balance to consider is $10,000
    min_balance = 2000              # Minimum balance to have always is $2,000
    balance = initial_balance   
    btc_held = 0                    # Initital coins held is 0
    num_buy_installments = 2        # Divide buying into 2 installments
    num_sell_installments = 2       # Divide selling into 2 installments
    transaction_fee_rate = 0.002    # 0.2% Transaction Fee
    trade_returns = []              # Initialize the trade returns

    # Loop through each chunk, where each chunk represents a time window (e.g., 15 min, 1 hr, etc.)
    for chunk, group in df.groupby('chunk'):
        # Reset the number of buy and sell installments for each new chunk
        num_buy_installments = 2
        num_sell_installments = 2

        # Iterate over each row in the current chunk
        for i in range(len(group)):
            # Check if it's not the last row, a buy signal is triggered, sufficient balance is available, and there are remaining buy installments
            if (i != (len(group)-1)) and buy_signal(group.iloc[i]) and balance > 2000 and num_buy_installments > 0:
                # Buy logic
                # Calculate the amount to invest for this installment
                amount_to_invest = (balance - min_balance) / num_buy_installments       
                
                # Calculate the amount of BTC to buy, accounting for transaction fees
                btc_to_buy = (amount_to_invest / group['close'].iloc[i]) * (1 - transaction_fee_rate)   
                # Update the BTC held and the remaining balance after the purchase
                btc_held += btc_to_buy
                balance -= amount_to_invest

                # Record the buy price for future sell logic
                buy_price = group['close'].iloc[i]

                # Update the buy and sell installments
                num_buy_installments -= 1   # As a buy is done we only have one more buy left
                num_sell_installments = 2   # As a buy is done, we can again do 2 sells
                # print(f"Buying Cryptocoin at {group['close'].iloc[i]:.2f} in chunk {chunk}, BTC held: {btc_held:.6f}, Balance: {balance:.2f}")

            # # Check if it's not the last row, a sell signal is triggered, BTC is held, and there are remaining sell installments
            elif (i >= 0 and i < (len(group)-1)) and sell_signal(group.iloc[i]) and btc_held > 0 and num_sell_installments > 0:
                # Sell logic
                # Calculate the amount of BTC to sell in this installment
                btc_to_sell = btc_held / num_sell_installments

                # Calculate the amount received from selling, accounting for transaction fees
                amount_to_receive = (btc_to_sell * group['close'].iloc[i]) * (1 - transaction_fee_rate)

                # Update the balance and reduce the BTC held after the sale
                balance += amount_to_receive
                btc_held -= btc_to_sell

                # Calculate the percentage return for this trade and store it
                trade_return = ((group['close'].iloc[i] - buy_price) / buy_price) * 100
                trade_returns.append(trade_return)

                # Update the buy and sell installments
                num_sell_installments -= 1
                num_buy_installments = 2
                # print(f"Selling Cryptocoin at {group['close'].iloc[i]:.2f} in chunk {chunk}, Balance: {balance:.2f}, Trade Return: {trade_return:.2f}%")

            # If it's the last row in the chunk and BTC is still held, sell all remaining BTC
            elif (i == (len(group)-1)) and btc_held > 0:
                # Sell logic
                # Sell all remaining BTC and calculate the amount received
                amount_to_receive = btc_held * group['close'].iloc[i] * (1 - 0.002)

                # Update the balance and reset BTC held to zero
                balance += amount_to_receive
                btc_held = 0

                # Calculate the percentage return for this trade and store it
                trade_return = ((group['close'].iloc[i] - buy_price) / buy_price) * 100
                trade_returns.append(trade_return)
                # print(f"Selling Cryptocoin at {group['close'].iloc[i]:.2f} in chunk {chunk}, Balance: {balance:.2f}, Trade Return: {trade_return:.2f}%, at: {i}")
                
            # Track balance, BTC held, and portfolio value at each step
            df.at[(group.index[i]), 'balance'] = balance
            df.at[(group.index[i]), 'btc_held'] = btc_held
            df.at[(group.index[i]), 'portfolio_value'] = balance + (btc_held * group['close'].iloc[i])  # Portfolio = cash + BTC value
    
    # Fill initial balance for rows where no action occurred
    df['balance'] = df['balance'].ffill()
    df['btc_held'] = df['btc_held'].ffill()
    df['portfolio_value'] = df['portfolio_value'].ffill()

    # Ensure the date column is in datetime format and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Metrics calculation
    annualized_return = ann_ret(df)                                 # Calculate the Annualized Return
    annualized_volatility = ann_vol(df)                             # Calculate the Annualized Volatility
    sharpe_ratio = sr(annualized_return, annualized_volatility)     # Calculate the Sharpe Ratio
    performance_ratio = pr(df, annualized_return)                   # Calculate the Performance Ratio

    # Reset the index for further processing
    df.reset_index(inplace=True)

    # Return the calculated metrics as a dictionary
    return {
        "Annualized Return": (annualized_return * 100), # Convert to percentage
        "Annualized Volatility": (annualized_volatility * 100), # Convert to percentage
        "Sharpe Ratio": sharpe_ratio,
        "Performance Ratio": performance_ratio
    }


# The code starts here

# Load a CSV file into a DataFrame
df = pd.read_csv("/root/Capstone Signalling/datasets/Data/BTC_training_data_2023-06-01_to_2023-11-30.csv")

# Sort the dataframe by date
df = df.sort_values('date')

# Display the first few groups of the dataset
print(df.head())

# Add all the indicators required for code
# Define EMA indicators
df['5_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
df['10_EMA'] = df['close'].ewm(span=10, adjust=False).mean()
df['20_EMA'] = df['close'].ewm(span=20, adjust=False).mean()
df['50_EMA'] = df['close'].ewm(span=50, adjust=False).mean()
df['100_EMA'] = df['close'].ewm(span=100, adjust=False).mean()
df['200_EMA'] = df['close'].ewm(span=200, adjust=False).mean()

#Define SMA indicators with their Standard Deviation
df['5_SMA'] = df['close'].rolling(window=5).mean()

df['20_SMA'] = df['close'].rolling(window=20).mean()
df['stddev20'] = df['close'].rolling(window=20).std()

df['50_SMA'] = df['close'].rolling(window=50).mean()
df['stddev50'] = df['close'].rolling(window=50).std()

df['100_SMA'] = df['close'].rolling(window=100).mean()
df['200_SMA'] = df['close'].rolling(window=200).mean()
df['250_SMA'] = df['close'].rolling(window=250).mean()
df['300_SMA'] = df['close'].rolling(window=300).mean()

# Define Bollinger Bands
df['upper_band'] = df['20_SMA'] + (df['stddev20'] * 2)
df['lower_band'] = df['20_SMA'] - (df['stddev20'] * 2)

df['upper_band50'] = df['50_SMA'] + (df['stddev50'] * 2)
df['lower_band50'] = df['50_SMA'] - (df['stddev50'] * 2)

# Define RSI
delta = df['close'].diff(1)
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

# Define window periods
window_periods = [15, 60, 240, 480, 1440, 10080]  # 15 min, 1 hr, 4 hr, 8 hr, 1 day, 1 week

# Run strategies across all strategies and all window periods
# Initailize the results
results = []
for strategy in strategies:                 # Run strategy by strategy
    for window_period in window_periods:    # For each strategy, run window by window
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
    print("done with strategy")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Group by window period and find the best strategy for each by checking the best Sharpe Ratio among the strategies
best_strategies_per_window = results_df.loc[
    results_df.groupby('window_period')['Sharpe Ratio'].idxmax()
]

# Display the results for each window
print("Best Strategy for Each Window:")
print(best_strategies_per_window)

# Optional: Iterate through the results and display more readable output
for _, row in best_strategies_per_window.iterrows():
    print(
        f"Best Strategy for {row['window_period']} minute window: "
        f"Strategy={row['strategy']}, Sharpe Ratio={row['Sharpe Ratio']:.2f}, "
        f"Annualized Return={row['Annualized Return']}%, "
        f"Annualized Volatility={row['Annualized Volatility']}%, "
        f"Performance Ratio={row['Performance Ratio']:.2f}"
    )