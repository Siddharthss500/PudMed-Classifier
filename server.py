import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
from scipy import sparse as sp_sparse
from nltk.corpus import stopwords
from flask import Flask, jsonify, request

app = Flask(__name__)


def pre_process(text):
    symbols = re.compile('[^A-Za-z0-9(),!?\'\`]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.replace('\n', ' ').lower()
    text = symbols.sub(' ', text)
    X = ' '.join([w for w in text.split() if not w in STOPWORDS])
    return [X]


def BOW(text, words_to_index, dict_size):
    # Simple BOW model
    result_vector = np.zeros(dict_size)
    for word in text.split(' '):
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def load_model(X):
    dict_size = 10000
    with open('pkl_file/words_to_idx.pickle', 'rb') as handle:
        words_to_idx = pickle.load(handle)
    X = sp_sparse.vstack([sp_sparse.csr_matrix(BOW(text, words_to_idx, dict_size)) for text in X])

    filename = 'models/BOW_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(X)
    return result


# @app.get("/")
@app.route("/predict", methods=['POST'])
async def get_prediction():
    abstract = request.get_json()
    data = abstract['data']
    data = pre_process(data)
    print("Data preprocessed")
    output = load_model(data)
    print("Model loaded")
    final_output = {"output": output}
    return jsonify(final_output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
