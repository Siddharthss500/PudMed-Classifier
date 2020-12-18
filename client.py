import requests
import pandas as pd

url = 'http://localhost:80/predict'


def main():
    df = pd.read_csv('data/test_df.csv')
    for abst in df['Abstract']:
        my_data = {'param': str(abst)}
        output = requests.post(url, json=my_data)
        print(output.json())


if __name__ == "__main__":
    main()
