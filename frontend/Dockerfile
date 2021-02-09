FROM python:3.6

WORKDIR /app

RUN apt-get update

#ARG CHROME_VERSION=88.0.4324.150
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable
#RUN apt-get install -y google-chrome=$CHROME_VERSION

ARG CHROME_DRIVER_VERSION=88.0.4324.96
#ARG CHROME_DRIVER_VERSION=89.0.4389.23
#ARG CHROME_DRIVER_VERSION=87.0.4280.88
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

#ENV DISPLAY=:99

EXPOSE 8501

CMD ["streamlit", "run", "frontend_main.py"]