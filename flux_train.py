import json
import requests
import numpy as np
from datetime import datetime
import time
import os
import logging

# Configure logging
logging.basicConfig(filename='data_collection.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to save training vector to a JSON file
def save_training_vector(data, filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                existing_data = json.load(file)
            existing_data.append(data)
            with open(filename, 'w') as file:
                json.dump(existing_data, file, indent=4)
        else:
            with open(filename, 'w') as file:
                json.dump([data], file, indent=4)
    except Exception as e:
        logging.error(f"Error occurred while saving training vector to {filename}: {e}")

# Function to fetch trade history and order book data from Trade Ogre API
def fetch_data(pair):
    try:
        trade_data = requests.get(f'https://tradeogre.com/api/v1/history/{pair}')
        order_data = requests.get(f'https://tradeogre.com/api/v1/orders/{pair}')
        
        if trade_data.status_code != 200 or order_data.status_code != 200:
            logging.error(f"Failed to fetch data from Trade Ogre API: Status Code - Trade: {trade_data.status_code}, Orders: {order_data.status_code}")
            return None, None
        
        trades = json.loads(trade_data.text)
        orders = json.loads(order_data.text)
        
        return trades, orders
    except Exception as e:
        logging.error(f"Error occurred while fetching data from Trade Ogre API: {e}")
        return None, None

# Initialize a variable to store the current UTC date
current_utc_date = datetime.utcnow().strftime("%Y-%m-%d")
# Initialize a set to keep track of processed trades
processed_trades = set()
while True:
    try:
        # Check if the current date has changed
        if datetime.utcnow().strftime("%Y-%m-%d") != current_utc_date:
            # If the date has changed, update the current date and reset the processed trades set
            current_utc_date = datetime.utcnow().strftime("%Y-%m-%d")
            processed_trades = set()

        # Fetch trade and order book data
        trades, orders = fetch_data('FLUX-USDT')

        if trades is None or orders is None:
            raise Exception("Failed to fetch data from Trade Ogre API")

        # Calculate VWAP
        prices = np.array([float(trade['price']) for trade in trades])
        volumes = np.array([float(trade['quantity']) for trade in trades])
        vwap = np.average(prices, weights=volumes)

        # Check for new trades occurred today
        for trade in trades:
            timestamp = trade['date']
            trade_date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    
            # Check if the trade occurred today
            if trade_date == current_utc_date:
                price = float(trade['price'])
                quantity = float(trade['quantity'])
                # This allows us to to create a composite key to ensure tuples with duplicate timestamps get logged
                trade_id = f"{timestamp}_{price}_{quantity}"  # Concatenate date, price, and quantity
        
                if trade_id not in processed_trades:
                    # Extract features from order book (e.g., bid-ask spread)
                    buy_prices = list(orders['buy'].keys())
                    buy_prices = np.float_(buy_prices)
                    sell_prices = list(orders['sell'].keys())
                    sell_prices = np.float_(sell_prices)
                    bid_ask_spread = buy_prices[0] - sell_prices[0]
            
                    # Combine features into a feature vector
                    feature_vector = {'timestamp': timestamp, 'price': price, 'qty': quantity,
                                    'vwap': vwap, 'bid_ask_spread': bid_ask_spread}

                    # Generate filename with UTC date
                    filename = f"FLUX-USDT_{current_utc_date}.json"

                    # Save training vector to JSON file
                    save_training_vector(feature_vector, filename)

                    # Mark trade as processed
                    processed_trades.add(trade_id)

                    print(f"New trade processed and saved to {filename}")

        # Wait for a short interval before checking for new trades again
        time.sleep(2)  # Adjust the interval as needed

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        time.sleep(15)  # Wait before retrying