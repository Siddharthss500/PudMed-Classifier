import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
from scipy import sparse as sp_sparse
from nltk.corpus import stopwords
from flask import Flask, jsonify, request
import logging
from sklearn import linear_model

app = Flask(__name__)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.DEBUG)


def BOW(text, words_to_index, dict_size):
    # Simple BOW model
    result_vector = np.zeros(dict_size)
    for word in text.split(' '):
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def pre_process(text):
    symbols = re.compile('[^A-Za-z0-9(),!?\'\`]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.replace('\n', ' ').lower()
    text = symbols.sub(' ', text)
    X = ' '.join([w for w in text.split() if not w in STOPWORDS])
    X = [X]
    dict_size = 10000
    with open('pkl_file/words_to_idx.pickle', 'rb') as handle:
        words_to_idx = pickle.load(handle)
    X = sp_sparse.vstack([sp_sparse.csr_matrix(BOW(text, words_to_idx, dict_size)) for text in X])
    return X


def load_model():
    filename = 'models/BOW_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


loaded_model = load_model()


def assign_class(output):
    output = tuple(output[0])
    out = {
        (0, 0): "Others",
        (0, 1): "Congenital Anomalies",
        (1, 0): "Drug Adverse Effects",
        (1, 1): "Both"
    }

    return out[output]


@app.route("/predict", methods=['POST'])
def get_prediction():
    abstract = request.get_json()
    app.logger.debug(f'Get Data:{abstract}')
    data = abstract['param']
    app.logger.debug(f'View data{data}')
    data = pre_process(data)
    app.logger.debug(f'Processed Data:{data}')
    output = loaded_model.predict(data)
    app.logger.debug(f'Output:{output}')
    final_output = {"output": assign_class(output)}
    return jsonify(final_output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)