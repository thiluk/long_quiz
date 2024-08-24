FROM python:3.10
ADD . .
WORKDIR /houseprice_api
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080
EXPOSE 8001
CMD ["python","app/main.py"]