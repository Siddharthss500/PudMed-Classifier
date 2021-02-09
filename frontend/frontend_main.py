import streamlit as st
import pandas as pd
from time import time
import requests
import DataDownload

# Uncomment the below line if you are building a docker image
url = 'http://backend:8080/predict'

# Uncomment the below line if using locally without dockerizing
# url = 'http://localhost:8080/predict'


@st.cache
def create_dataset(two_entry_sets, num=20):
    return DataDownload.create_dataset(two_entry_sets, num)


@st.cache
def entry_terms_create(two_entry_sets, search_term_one, search_term_two):
    return pd.DataFrame(data=two_entry_sets[0], columns=[search_term_one]), \
           pd.DataFrame(data=two_entry_sets[1], columns=[search_term_two])


@st.cache
def entry_terms_extractor(search_terms):
    return DataDownload.create_entry_terms(search_terms)


def test_dataset():
    test_df = pd.read_csv('data/test_df.csv')
    test_df = test_df.sample(n=10)
    actual_output = [assign_class(i) for i in list(zip(test_df.term_one, test_df.term_two))]
    output_df = pd.DataFrame(test_df.Abstract, columns=["Abstract"])
    output_df["Actual Output"] = actual_output
    return output_df


# Assign classes based on the output vector
def assign_class(output):
    out = {
        (0, 0): "Others",
        (0, 1): "Congenital Anomalies",
        (1, 0): "Drug Adverse Effects",
        (1, 1): "Both"
    }
    return out[output]


def main():
    # Title
    st.title("Pub-Med Classifier")

    st.header("Data Creation")
    st.write("Dataset is created based on two search terms given by user. Note that the search term has to be present in the MeSH database (verbatim).")
    st.write("Note that the app takes sometime only when it is loaded for the first time. If the search terms do not change, then loading time increases drastically.")
    # Search terms
    st.subheader("Search terms")
    # User to enter two search terms
    search_term_one = st.text_input("Please enter search term 1", value="adverse drug events")
    search_term_two = st.text_input("Please enter search term 2", value="abnormalities, congenital")
    # Search terms = adverse drug events, abnormalities, congenital
    search_terms = [search_term_one, search_term_two]

    if st.button("Create Dataset"):
        with st.spinner("Extracting Entry Terms..."):
            two_entry_sets = entry_terms_extractor(search_terms)
            # Show the entry terms
            st.subheader("Entry terms")
            col1, col2 = st.beta_columns(2)
            search_term_one_df, search_term_two_df = entry_terms_create(two_entry_sets, search_term_one, search_term_two)
            with col1:
                st.write(search_term_one_df)
            with col2:
                st.write(search_term_two_df)

        # Create dataset
        with st.spinner("Creating dataset..."):
            start_time = time()
            answer = create_dataset(two_entry_sets, 20)
            total_time = time() - start_time
            if answer == "Done":
                st.subheader("Sample Data")
                st.write("Time taken to create the dataset is ", total_time, "seconds.")
                df = pd.read_csv("final_df.csv")
                st.write(df.sample(n=5, random_state=42))

    else:
        two_entry_sets = entry_terms_extractor(search_terms)
        with st.spinner("Loading Entry Terms..."):
            # Show the entry terms
            st.subheader("Entry terms")
            col1, col2 = st.beta_columns(2)
            search_term_one_df, search_term_two_df = entry_terms_create(two_entry_sets, search_term_one,
                                                                        search_term_two)
            with col1:
                st.write(search_term_one_df)
            with col2:
                st.write(search_term_two_df)

        # Create dataset
        with st.spinner("Creating dataset..."):
            start_time = time()
            answer = create_dataset(two_entry_sets, 20)
            total_time = time() - start_time
            if answer == "Done":
                st.subheader("Sample Data")
                st.write("Time taken to create the dataset is ", total_time, "seconds.")
                df = pd.read_csv("final_df.csv")
                st.write(df.sample(n=5, random_state=42))

    # Prediction
    st.header("Data Prediction")
    st.write("Ten random data points are taken and classified into one of four classes based on the algorithm selected.")
    model_name = st.selectbox("Choose model to predict", ["BOW", "tf-idf", "LSTM"])

    if st.button("Predict"):
        output_df = test_dataset()
        all_outputs = list()
        with st.spinner("Making Predictions..."):
            pred_output = list()
            # Send every abstract serially
            for abst in output_df['Abstract']:
                my_data = {'param': str(abst), 'method': model_name}
                pred_output = requests.post(url, json=my_data)
                all_outputs.append(pred_output.json()["output"])
            output_df["Predicted Output"] = all_outputs
            st.write(output_df)


if __name__ == "__main__":
    main()
