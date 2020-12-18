import requests
import pandas as pd

url = 'http://localhost:80/predict'


def main():
    # Load file
    df = pd.read_csv('data/test_df.csv')
    # Send every abstract serially
    for abst in df['Abstract']:
        my_data = {'param': str(abst)}
        # Send a request using REST API
        output = requests.post(url, json=my_data)
        # Print the class of the abstract
        print(output.json())


if __name__ == "__main__":
    main()
