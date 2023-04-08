from typing import Dict, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
)

from nlu.utils import get_db_intents, slot_uniformization

# Intent Classifier
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# NER Extractor
model_name = "qanastek/XLMRoberta-Alexa-Intents-NER-NLU"
tokenizer_ner = AutoTokenizer.from_pretrained(model_name)
model_ner = AutoModelForTokenClassification.from_pretrained(model_name)
predict_ner = TokenClassificationPipeline(model=model_ner, tokenizer=tokenizer_ner)


def debug_user_intent(
    user_input: str, intents_list: Dict[str, str]
) -> Tuple[str, float]:
    models = [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
    ]
    for model_name in models:
        print(f"======= Model: {model_name} =======")

        model = SentenceTransformer(model_name)
        intent_embeddings = {
            intent: model.encode(utterances["utterances"], convert_to_tensor=True)
            for intent, utterances in intents_list.items()
        }

        user_input_embedding = model.encode(user_input, convert_to_tensor=True)
        cosine_similarities = {}
        for intent in intent_embeddings:
            similarity = util.cos_sim(user_input_embedding, intent_embeddings[intent])
            print(f"{intent} max: {torch.max(similarity)}")
            print(f"{intent} mean: {torch.mean(similarity)}")
            cosine_similarities[intent] = (
                torch.max(similarity) * 2 + torch.mean(similarity)
            ) / 3
        most_similar_intent = max(cosine_similarities, key=cosine_similarities.get)
        print(f"{most_similar_intent} {max(cosine_similarities.values())}")
        print("=====================================")
    return most_similar_intent, max(cosine_similarities.values())


async def get_user_intent(user_input: str) -> Tuple[str, float]:
    intents_list = await get_db_intents()
    intent_embeddings = {
        intent: model.encode(utterances["utterances"], convert_to_tensor=True)
        for intent, utterances in intents_list.items()
    }

    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_similarities = {}
    for intent in intent_embeddings:
        similarity = util.cos_sim(user_input_embedding, intent_embeddings[intent])
        cosine_similarities[intent] = (
            torch.max(similarity) * 3 + torch.mean(similarity)
        ) / 4
    most_similar_intent = max(cosine_similarities, key=cosine_similarities.get)
    return most_similar_intent, max(cosine_similarities.values())


def slot_filling(user_input: str) -> str:
    ner = predict_ner(user_input)
    return slot_uniformization(ner)
