import requests
import pandas as pd
import datetime

def get_dataset(start_date, end_date):
    cryptoCurrencies = ["BTC", "ETH", "LTC"]
    for cryptoCurrency in cryptoCurrencies:
        symbol = cryptoCurrency
        currency = 'INR'
        time_difference = end_date - start_date

        # Extract the number of days from the time difference
        number_of_days = time_difference.days
        end_timestamp = int(end_date.timestamp())

        api_url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={currency}&limit={number_of_days}&toTs={end_timestamp}&api_key=API KEY'

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()['Data']['Data']
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            df.to_csv(f'{symbol}_historical_data.csv')

