FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y curl

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN mv /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

ENV PYTHONUNBUFFERED 1

COPY poetry.lock pyproject.toml README.md /app/

COPY ./nlu nlu

RUN poetry config virtualenvs.create false \
  && poetry install

RUN --mount=type=secret,id=HUGGING_FACE_HUB_TOKEN \
  export HUGGING_FACE_HUB_TOKEN=$(cat /run/secrets/HUGGING_FACE_HUB_TOKEN) \
  && python nlu/load_models.py

EXPOSE 8080

CMD ["uvicorn", "nlu.app:app", "--host", "0.0.0.0", "--port", "8080"]
