# lightweight and fast
FROM python:3.9-slim

# set containers working dir
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cp everything in
# COPY . .
# copy everything in souce into app working dir
COPY ./ /app

# Make port 5000 available to the host
EXPOSE 5000

# go
ENTRYPOINT [ "python" ]
CMD ["app.py"]

