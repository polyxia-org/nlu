from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

# Intent classifier
model_name_intent = "joeddav/xlm-roberta-large-xnli"
intent_classifier = pipeline("zero-shot-classification", model=model_name_intent)

# Sentence similarity
model_name_sim = "paraphrase-multilingual-mpnet-base-v2"
model_sentence_similarity = SentenceTransformer(model_name_sim)

# NER Extractor
model_name_ner = "qanastek/XLMRoberta-Alexa-Intents-NER-NLU"
tokenizer_ner = AutoTokenizer.from_pretrained(model_name_ner)
model_ner = AutoModelForTokenClassification.from_pretrained(model_name_ner)
predict_ner = TokenClassificationPipeline(model=model_ner, tokenizer=tokenizer_ner)
