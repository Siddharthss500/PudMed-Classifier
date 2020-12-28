import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
from scipy import sparse as sp_sparse
from nltk.corpus import stopwords
from flask import Flask, jsonify, request
import logging
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras.backend as K
from sklearn import linear_model

# Create Flask Application
app = Flask(__name__)

# Setup to show logging while code is running on the docker image
gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.DEBUG)


# Note : This function was taken from medium : https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# Pre-load all the models
# BOW model
def load_BOW_model():
    filename = 'models/BOW_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


# tfidf model
def load_tfidf_model():
    filename = 'models/tfidf_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


# LSTM model
def load_LSTM_model():
    filename = 'models/LSTM_model.h5'
    loaded_model = load_model('models/LSTM_model.h5', custom_objects={"get_f1": get_f1})
    return loaded_model


# Load all the models
loaded_model_BOW = load_BOW_model()
loaded_model_tfidf = load_tfidf_model()
loaded_model_LSTM = load_LSTM_model()


# BOW vectorizer
def BOW_vectorizer(text, words_to_index, dict_size):
    # Simple BOW model
    result_vector = np.zeros(dict_size)
    for word in text.split(' '):
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


# Simple pro-processing for text
def pre_process(text):
    symbols = re.compile('[^A-Za-z0-9(),!?\'\`]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.replace('\n', ' ').lower()
    text = symbols.sub(' ', text)
    X = ' '.join([w for w in text.split() if not w in STOPWORDS])
    return X


# BOW model
def BOW(X):
    X = [X]
    dict_size = 10000
    with open('pkl_file/words_to_idx.pickle', 'rb') as handle:
        words_to_idx = pickle.load(handle)
    # Build feature matrix
    X = sp_sparse.vstack([sp_sparse.csr_matrix(BOW_vectorizer(text, words_to_idx, dict_size)) for text in X])
    return X


# tfidf model
def tfidf(X):
    X = [X]
    # Load the vectorizer
    with open('pkl_file/tfidfvectorizer.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)
    return vectorizer.transform(X)


# LSTM model
def LSTM_model(X):
    # Load the tokenizer
    with open('pkl_file/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    maxlen = 200
    # Tokenize
    X = tokenizer.texts_to_sequences(X)
    # Pad sequences
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    return X


# Assign classes based on the output vector
def assign_class(output, method):
    if method == 'LSTM':
        class_val = []
        if output[0][0] > 0.5:
            class_val.append(1)
        else:
            class_val.append(0)
        if output[0][1] > 0.5:
            class_val.append(1)
        else:
            class_val.append(0)
        output = tuple(class_val)
    else:
        output = tuple(output[0])
    out = {
        (0, 0): "Others",
        (0, 1): "Congenital Anomalies",
        (1, 0): "Drug Adverse Effects",
        (1, 1): "Both"
    }

    return out[output]


# Main POST that is listening for any input
@app.route("/predict", methods=['POST'])
def get_prediction():
    # Get data in JSON format
    abstract = request.get_json()
    app.logger.debug(f'Get Data:{abstract}')
    # Extract the abstract text
    data = abstract['param']
    app.logger.debug(f'View Data{data}')
    # Pre-process the data
    data = pre_process(data)
    app.logger.debug(f'Processed Data:{data}')
    # Extract the method to classify
    method = abstract['method']
    app.logger.debug(f'Output:{method}')
    if method == 'BOW':
        data = BOW(data)
        # Run it through the trained BOW model
        output = loaded_model_BOW.predict(data)
        app.logger.debug(f'Output:{output}')
    elif method == 'tf-idf':
        data = tfidf(data)
        # Run it through the trained tfidf model
        output = loaded_model_tfidf.predict(data)
        app.logger.debug(f'Output:{output}')
    elif method == 'LSTM':
        data = LSTM_model(data)
        # Run it through the trained tfidf model
        output = loaded_model_LSTM.predict(data)
        app.logger.debug(f'Output:{output}')
    # Get the corresponding class of the output vector
    final_output = {"output": assign_class(output, method)}
    return jsonify(final_output)


# Flask app runs on port 80
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)