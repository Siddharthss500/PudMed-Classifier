import streamlit as st
import pandas as pd
from time import time
import requests
import DataDownload

url = 'http://backend:8080/predict'

def main():
    # Title
    st.title("Pub-Med Classifier")

    # Search terms
    st.header("Search terms")
    # User to enter two search terms
    search_term_one = st.text_input("Please enter search term 1", value="adverse drug events")
    search_term_two = st.text_input("Please enter search term 2", value="abnormalities, congenital")
    # Search terms = adverse drug events, abnormalities, congenital
    search_terms = [search_term_one, search_term_two]

    if st.button("Create Dataset"):
        two_entry_sets = DataDownload.create_entry_terms(search_terms)
        # Show the entry terms
        st.header("Entry terms")
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write(pd.DataFrame(data=two_entry_sets[0], columns=[search_term_one]))
        with col2:
            st.write(pd.DataFrame(data=two_entry_sets[1], columns=[search_term_two]))

        # Create dataset
        start_time = time()
        answer = DataDownload.create_dataset(two_entry_sets, 20)
        total_time = time() - start_time
        if answer == "Done":
            st.header("Sample Data")
            st.write("Time taken to create the dataset is ", total_time, "seconds.")
            df = pd.read_csv("final_df.csv")
            st.write(df.sample(n=5, random_state=42))

        # Prediction
        st.header("Prediction")
        model_name = st.selectbox("Choose model to predict", ["BOW", "tf-idf", "LSTM"])

        if st.button("Predict"):
            test_df = pd.read_csv('frontend/data/test_df.csv')
            if model_name == "BOW":
                # Send every abstract serially
                for abst in test_df['Abstract']:
                    my_data = {'param': str(abst), 'method': model_name}
                    st.write(requests.post(url, json=my_data))
            elif model_name == "tf-idf":
                # Send every abstract serially
                for abst in test_df['Abstract']:
                    my_data = {'param': str(abst), 'method': model_name}
                    st.write(requests.post(url, json=my_data))
            elif model_name == "LSTM":
                # Send every abstract serially
                for abst in test_df['Abstract']:
                    my_data = {'param': str(abst), 'method': model_name}
                    st.write(requests.post(url, json=my_data))


if __name__ == "__main__":
    main()
