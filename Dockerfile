# Dockerfile

FROM python:3.11

COPY . /app
WORKDIR /app
RUN pip install -e .

CMD ["python", "nexus/loomnode/main.py"]