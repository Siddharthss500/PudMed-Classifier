# PudMed-Classifier

The aim is to deploy a simple ML model (Bag-Of-Words Classifer Model) for Multi-Label Classification. A simple UI is built using Streamlit with FASTAPI in the backend. Docker is used to containerize FASTAPI and Streamlit. The application is deployed on an EC2 instance on AWS.

## File structure

```
├── backend
│   ├── models
│   │   ├── BOW_model.sav
│   │   ├── LSTM_model.h5
│   │   ├── tfidf_model.sav
│   ├── pkl_file
│   │   ├── tfidfvectorizer.pickle
│   │   ├── tokenizer.pickle
│   │   ├── words_to_idx.pickle
│   ├── backend.py
│   ├── Dockerfile
│   ├── models_inference.py
│   └── requirements.txt
├── frontend
│   ├── data
│   │   ├── test_df.csv
│   ├── DataDownload.py
│   ├── Dockerfile
│   ├── final_df.csv
│   ├── frontend.py
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```

## How to run locally

```
docker-compose up -d --build
```

Note that before running the above command the repository must be cloned and a few lines of code should be commented/uncommented (as mentioned in the commments of DataDownload.py and frontend.py). The command will create a docker container that connects two docker images (the frontend to the backend). The UI will show up on   http://localhost:8501.


The frontend and backend can be tested separately. To test them separately we run the following commands, 

#### To test frontend
```
docker build -t frontend .
```

```
docker run -p 8501:8501 frontend
``` 
#### To test backend
```
docker build -t backend .
```

```
docker run -p 8080:8080 backend
``` 

Note that the streamlit UI is setup on the port 8501 and FAST API listens on the port 8080. 

## Future work

1. Improve currently deployed ML/DL models
2. Integrate model training into UI and make it a standalone application





