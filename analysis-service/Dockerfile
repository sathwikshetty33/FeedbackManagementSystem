FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

ENV PORT=8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
