import time
from typing import Dict, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

from nlu.crud.skills import CRUDSkills
from nlu.utils import get_db_intents, slot_uniformization
import logging 

logger = logging.getLogger(__name__)

#### Intent Classifier
# Sentence-BERT: https://arxiv.org/pdf/1908.10084.pdf
# https://github.com/microsoft/MPNet
model_sentence_similarity = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# XLM-RoBERTa: https://arxiv.org/pdf/1911.02116.pdf
intent_classifier = pipeline(
    "zero-shot-classification", model="joeddav/xlm-roberta-large-xnli"
)

#### NER Extractor
model_name_ner = "qanastek/XLMRoberta-Alexa-Intents-NER-NLU"
tokenizer_ner = AutoTokenizer.from_pretrained(model_name_ner, local_files_only=True)
model_ner = AutoModelForTokenClassification.from_pretrained(
    model_name_ner, local_files_only=True
)
predict_ner = TokenClassificationPipeline(model=model_ner, tokenizer=tokenizer_ner)


def debug_user_intent(
    user_input: str, intents_list: Dict[str, str]
) -> Tuple[str, float]:
    print("======= Model sentence similarity =======")
    start_time = time.perf_counter()
    intent_embeddings = {
        intent: model_sentence_similarity.encode(
            utterances["utterances"], convert_to_tensor=True
        )
        for intent, utterances in intents_list.items()
    }
    user_input_embedding = model_sentence_similarity.encode(
        user_input, convert_to_tensor=True
    )
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
    print(f"sentence similarity time: {time.perf_counter() - start_time}")
    print("=====================================")

    print("======= Model intent classifier =======")
    start_time = time.perf_counter()
    candidate_labels = list(intent_embeddings.keys())
    intent_classified = intent_classifier(
        user_input, candidate_labels, multi_label=True
    )
    for idx, intent in enumerate(intent_classified["labels"]):
        print(f"{intent} {intent_classified['scores'][idx]}")
    print(f"intent classifier time: {time.perf_counter() - start_time}")
    print("=====================================")

    if intent_classified["scores"][0] > 0.81:
        return intent_classified["labels"][0], intent_classified["scores"][0]
    elif max(cosine_similarities.values()) > 0.72:
        return most_similar_intent, max(cosine_similarities.values())

    return None, None


async def get_user_intent(user_input: str) -> Tuple[str, float]:
    intents_list = await get_db_intents()

    # zero-shot classification
    candidate_labels = list(intents_list.keys())
    if not candidate_labels:
        return None, None
    intent_classified = intent_classifier(
        user_input, candidate_labels, multi_label=True
    )
    logger.info(f"Intent classifier {intent_classified['labels'][0]} {intent_classified['scores'][0]}")
    if intent_classified["scores"][0] > 0.81:        
        return intent_classified["labels"][0], intent_classified["scores"][0]

    # sentence similarity
    intent_embeddings = {
        intent: model_sentence_similarity.encode(
            utterances["utterances"], convert_to_tensor=True
        )
        for intent, utterances in intents_list.items()
    }

    user_input_embedding = model_sentence_similarity.encode(
        user_input, convert_to_tensor=True
    )
    cosine_similarities = {}
    for intent in intent_embeddings:
        similarity = util.cos_sim(user_input_embedding, intent_embeddings[intent])
        cosine_similarities[intent] = (
            torch.max(similarity) * 3 + torch.mean(similarity)
        ) / 4
    most_similar_intent = max(cosine_similarities, key=cosine_similarities.get)
    most_similar_intent_score = cosine_similarities[most_similar_intent]
    logger.info(f"Sentence similarity {most_similar_intent} {most_similar_intent_score}")
    if most_similar_intent_score > 0.72:        
        return most_similar_intent, most_similar_intent_score

    return None, None


def named_entity_recognition(user_input: str) -> str:
    ner = predict_ner(user_input)
    return slot_uniformization(ner)


async def get_params(intent: str, user_input: str):
    """Get parameters from user input"""
    skill = await CRUDSkills.get_by_name(intent)
    slots = skill[0].get("slots")
    if not slots:
        return {}
    ner = named_entity_recognition(user_input)
    slots = [slot["type"] for slot in slots]
    params = {}
    for slot_type, value in ner:
        if slot_type in slots:
            params[slot_type] = value
    return params
