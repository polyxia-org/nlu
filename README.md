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
poetry run python -m nltk.downloader all
```

## Usage

Train : 
```bash
poetry run python nlu/train.py nlu/intents
```

Use : 
```bash
# To try the NLU intent detection interactively
poetry python nlu/cli.py

#To use as a webapp
poetry run python nlu/app.py
```
