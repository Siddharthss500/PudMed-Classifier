 #Using the base image with python 3.6
 FROM python:3.6

 #Set our working directory as app
 COPY . /app
 WORKDIR /app
 #Installing python packages pandas, scikit-learn and gunicorn
# RUN pip install pandas scikit-learn flask gunicorn
 RUN pip install -r requirements.txt

 # Copy the models directory and server.py files
 ADD ./models ./models
 ADD ./pkl_file ./pkl_file
 ADD server.py server.py

 #Exposing the port 5000 from the container
 EXPOSE 80
 #Starting the python application
 CMD ["gunicorn", "--bind", "0.0.0.0:80", "server:app", "--log-level", "DEBUG"]
