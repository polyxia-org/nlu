def get_date():
    return "Today is the 16th March 2023"


def get_weather():
    return "It is 15 degrees and sunny"


def say_hello():
    return "Hello there!"


def order_tacos():
    return "Tacos are on the way!"


def light_on():
    return "Turning on the light!"


def chat_bot(chat_bot, text: str) -> str:
    return chat_bot.ask(text)
