FROM python:3.12

COPY . .

COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python","manage.py","runserver","0.0.0.0:8000"]
