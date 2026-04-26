FROM python:3.13-slim

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

ENTRYPOINT ["openfreqbench"]
