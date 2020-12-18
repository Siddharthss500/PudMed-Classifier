# PudMed-Classifier

The aim is to deploy a simple ML model (Bag-Of-Words Classifer Model) for Multi-Label Classification. Flask is used to build a simple REST API (to expose the API through web UI). Docker holds the ML model application. With this setup, one can invoke the ML model service using any REST client.

## File structure

*data* - It has a small data file that is used for classification <br>
*models* - Trained (not fully) ML classifier models are present. They are Bag-Of-Words model, tf-idf model, LSTM and BERT <br>
*pkl_file* - It contains intermediate pickle files that are required for data pre-processing based on the model <br>
*client.py* - It pings the *server.py* file serially to receive classified output <br>
*server.py* - Flask application that accepts data to predict classes and returns a JSON format <br>
*Dockerfile* - It is used to build the docker image <br>
*requirements.txt* - It has a list of library packages needed for the docker image

## How to run locally

```
docker build -t classifier-model .
```
```
docker run -p 80:80 classifier-model
```

Note that before running the above two commands the repository must be cloned. Once inside the folder the following commands will pull up the docker image. 
```
python client.py
```
This command will ping the docker and receive the classified outputs continously.

## Future work

1. Implement all models
2. Productionalize the code





