# lightweight and fast
# FROM python:3.9-slim
FROM python:3.13-slim

# bc running as a non-root, create a user that matches host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} coolcat && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash coolcat
USER coolcat

# set containers working dir
WORKDIR /app

COPY requirements.txt .
# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy everything in souce into app working dir
COPY ./ /app

# overcome permissions issues
USER root
RUN chown -R coolcat:coolcat /app
USER coolcat


# install the package in development mode
# editable mode allows you to modify the source code and see the changes reflected in the package without having to reinstall it
RUN pip install --editable .

# Set the entrypoint to python
ENTRYPOINT ["python"]

# docker build -t generalized-timeseries:latest ./
# docker run -it generalized-timeseries:latest /app/generalized_timeseries/examples/example.py
# -it for interactive, tty mode, allowing typing in the terminal, color formatting, etc

# to run the container in interactive mode, without using python as the entrypoint
# docker run -it --entrypoint /bin/bash generalized-timeseries:latest

