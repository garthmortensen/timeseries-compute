# lightweight and fast
# FROM python:3.9-slim
FROM python:3.13-slim

# bc running as a non-root, create a user that matches host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} timeseriesapp && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash timeseriesapp
USER timeseriesapp

# set containers working dir
WORKDIR /app

COPY requirements.txt .
# install dependencies
RUN pip install uv && \
    uv venv && \
    source .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    uv pip install --editable .

# copy everything in souce into app working dir
COPY ./ /app

# overcome permissions issues
USER root
RUN chown -R timeseriesapp:timeseriesapp /app
USER timeseriesapp


# install the package in development mode
# editable mode allows you to modify the source code and see the changes reflected in the package without having to reinstall it
RUN pip install --editable .

# Set the entrypoint to python
ENTRYPOINT ["python"]

# docker build -t timeseries-compute:latest ./
# docker run -it timeseries-compute:latest /app/timeseries_compute/examples/example.py
# -it for interactive, tty mode, allowing typing in the terminal, color formatting, etc

# to run the container in interactive mode, without using python as the entrypoint
# docker run -it --entrypoint /bin/bash timeseries-compute:latest

