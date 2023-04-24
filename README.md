# Polyxia NLU

This is a NLU (Natural Language Understanding) project. It's purpose is to understand the user's intent and extract entities from user's sentence.
To then call the corresponding action (morty functions or LLM chatbot in fallback if no intent is detected).

### Requirements

1. Install poetry

```bash
# If poetry is not installed
curl -sSL https://install.python-poetry.org | python3 -
```

2. Create an account on Hugging Face: https://huggingface.co/join?next=%2Fjoeddav%2Fxlm-roberta-large-xnli

3. Agree to share informations on the repo and create a token: https://huggingface.co/joeddav/xlm-roberta-large-xnli

4. Install Hugging Face CLI: `pip install --upgrade huggingface_hub`

   > https://huggingface.co/docs/huggingface_hub/quick-start

5. Login to Hugging Face: `huggingface-cli login` (token role is `read`)

## Installation

```bash
git clone  ...
cd nlu
poetry install
```

Rename .env.template file to .env and fill the values

## Usage

Use :

```bash
# To try the NLU intent detection interactively
poetry run python nlu/cli.py

#To use as a webapp (first launch a surrealdb instance)
docker run --rm --pull always -p 8000:8000 surrealdb/surrealdb:latest start --pass root
poetry run python nlu/app.py
```

### Run with Docker:

Using remote image :

```bash
docker compose up -d
```

Building locally (set your Hugging Face token) :

```bash
export HUGGING_FACE_HUB_TOKEN=<my_token>
docker buildx build --secret id=HUGGING_FACE_HUB_TOKEN -t ghcr.io/polyxia-org/nlu:latest .
docker compose up -d
```
