from nlu.brain import debug_user_intent, named_entity_recognition
from nlu.functions import INTENTS_HANDLER
from nlu.utils import get_mockup_intents

INTENTS = get_mockup_intents("nlu/intents")

print("Enter a query to get the associated intent (type 'quit' to exit)")
while True:
    payload = input("You: ")
    if payload == "quit":
        break

    intent, prob = debug_user_intent(payload, INTENTS)
    print(f"{intent} {prob}")

    if intent is not None:
        print(f"Detected intent: {intent}")
        print(
            {
                "text": payload,
                "language": "TBD",
                "intent_class": intent,
                "named_entities": named_entity_recognition(payload),
            }
        )
        function_to_call = INTENTS_HANDLER.get(intent)
        print(function_to_call)
        if function_to_call:
            print(function_to_call())
    else:
        print("Cannot find an intent that match your query")
