from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

# Intent Classifier
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# NER Extractor
model_name = 'qanastek/XLMRoberta-Alexa-Intents-NER-NLU'
tokenizer_ner = AutoTokenizer.from_pretrained(model_name)
model_ner = AutoModelForTokenClassification.from_pretrained(model_name)
predict_ner = TokenClassificationPipeline(model=model_ner, tokenizer=tokenizer_ner)

def get_user_intent(user_input: str, intents_list: Dict[str, str]) -> Tuple[str, float]:
    intent_embeddings = {
        intent: model.encode(utterances["utterances"])
        for intent, utterances in intents_list.items()
    }

    user_input_embedding = model.encode(user_input)
    cosine_similarities = {}
    for intent in intent_embeddings:
        similarity = cosine_similarity([user_input_embedding], intent_embeddings[intent])
        print(f"{intent} max: {np.max(similarity)}")
        print(f"{intent} mean: {np.mean(similarity)}")
        cosine_similarities[intent] = (np.max(similarity) + np.mean(similarity)) / 2
    most_similar_intent = max(cosine_similarities, key=cosine_similarities.get)
    print(f"{most_similar_intent} {max(cosine_similarities.values())}")
    return most_similar_intent, max(cosine_similarities.values())

def get_slots(user_input: str, intent: str) -> str:
    pass