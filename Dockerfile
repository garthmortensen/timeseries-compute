# lightweight and fast
FROM python:3.9-slim

# set containers working dir
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cp everything in
COPY . .

# create dir structure
RUN mkdir -p logs results
# just in case
RUN chmod -R 777 ./logs ./results

# go
ENTRYPOINT [ "python" ]
# CMD ["401k.py"]
