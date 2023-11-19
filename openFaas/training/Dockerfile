FROM python:3.8-alpine

# Install build dependencies for scipy
RUN apk --no-cache add musl-dev gcc gfortran lapack-dev

WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./training/handler.py"]
