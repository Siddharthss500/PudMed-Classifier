import requests
import pandas as pd

# Note : Before running the code, we need to clone this repository and run the docker image
# To build the docker image - docker build -t classifier-model . (if inside the required folder)
# To run the docker image - docker run -p 80:80 classifier-model
url = 'http://localhost:80/predict'


def main():
    # Load file
    df = pd.read_csv('data/test_df.csv')
    # Send every abstract serially
    for abst in df['Abstract']:
        my_data = {'param': str(abst), 'method': 'LSTM'}
        # Send a request using REST API
        output = requests.post(url, json=my_data)
        # Print the class of the abstract
        print(output.json())


if __name__ == "__main__":
    main()
