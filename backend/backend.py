import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import Any, Dict, AnyStr, Union
from fastapi.responses import JSONResponse

from models_inference import pre_process, BOW, loaded_model_BOW, loaded_model_tfidf, loaded_model_LSTM, tfidf, LSTM_model, assign_class

app = FastAPI()

JSONObject = Dict[AnyStr, Any]
JSONObject2 = Dict[AnyStr, Any]
JSONStructure = Union[JSONObject, JSONObject2]

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/predict")
def get_prediction(abstract: JSONStructure = None):
    abstract = jsonable_encoder(abstract)
    data = abstract['param']
    data = pre_process(data)
    method = abstract['method']
    if method == 'BOW':
        data = BOW(data)
        # Run it through the trained BOW model
        output = loaded_model_BOW.predict(data)
    elif method == 'tf-idf':
        data = tfidf(data)
        # Run it through the trained tfidf model
        output = loaded_model_tfidf.predict(data)
    elif method == 'LSTM':
        data = LSTM_model(data)
        # Run it through the trained tfidf model
        output = loaded_model_LSTM.predict(data)
    # Get the corresponding class of the output vector
    final_output = {"output": assign_class(output, method)}
    return JSONResponse(content=final_output)


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8080)
