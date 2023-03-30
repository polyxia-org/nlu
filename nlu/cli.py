from nlu.utils import get_intents
from nlu.brain import get_user_intent

from nlu.functions import INTENTS_HANDLER

INTENTS = get_intents("nlu/intents")

print("Enter a query to get the associated intent (type 'quit' to exit)")
while True:
    payload = input("You: ")
    if payload == "quit":
        break

    intent, prob = get_user_intent(payload, INTENTS)

    if prob > 0.75:
        print(f"Detected intent: {intent}")
        # print(
        #     {
        #         "text": payload,
        #         "language": "TBD",
        #         "intent_class": intent,
        #         "named_entities": named_entities,
        #     }
        # )
        print(INTENTS_HANDLER.get(intent)())
    else:
        print("Cannot find an intent that match your query")
