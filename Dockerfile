# lightweight and fast
FROM python:3.13-slim

# bc running as a non-root, create a user that matches host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} timeseriesapp && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash timeseriesapp

# set containers working dir
WORKDIR /app

# copy everything in source into app working dir first
COPY ./ /app

# Install uv as root for system-wide availability
USER root
RUN pip install uv
RUN chown -R timeseriesapp:timeseriesapp /app

# Switch to non-root user for dependency installation
USER timeseriesapp

# Install dependencies and package using uv
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e ".[dev]"

# Set the entrypoint to python
ENTRYPOINT ["python"]

# docker build -t timeseries-compute:latest ./
# docker run -it timeseries-compute:latest /app/timeseries_compute/examples/example.py
# -it for interactive, tty mode, allowing typing in the terminal, color formatting, etc

# to run the container in interactive mode, without using python as the entrypoint
# docker run -it --entrypoint /bin/bash timeseries-compute:latest

