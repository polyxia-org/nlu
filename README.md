# Polyxia NLU

This is a NLU (Natural Language Understanding) project. It's purpose is to understand the user's intent and extract entities from user's sentence.
To then call the corresponding action (morty functions or LLM chatbot in fallback if no intent is detected).

### Requirements

```bash
# If poetry is not installed
curl -sSL https://install.python-poetry.org | python3 -
```

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
