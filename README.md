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

Rename .env.template file to .env and fill the values

## Usage

Train : 
```bash
poetry run python nlu/train.py nlu/intents
```

Use : 
```bash
# To try the NLU intent detection interactively
poetry run python nlu/cli.py

#To use as a webapp
poetry run python nlu/app.py
```
Alternatively:
```bash
# You can also use poetry shell to use the virtualenv and run python commands directly
poetry shell
python nlu/cli.py
python nlu/app.py
```
