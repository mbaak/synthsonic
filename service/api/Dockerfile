FROM tiangolo/uvicorn-gunicorn:python3.7

ENV PYTHONPATH /app

COPY . /app/api/
RUN mv /app/api/*whl /tmp/

WORKDIR /app

RUN pip install -r /app/api/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]