FROM python:3.7-slim

RUN apt-get update
RUN apt-get install -y apt-file
RUN apt-file update
RUN apt-get install -y nano

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "ui.py"]