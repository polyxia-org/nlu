from nlu.functions import (
    chat_bot,
    get_date,
    get_weather,
    light_on,
    order_tacos,
    say_hello,
)

INTENTS_HANDLER = {
    "get_date": get_date,
    "get_weather": get_weather,
    "say_hello": say_hello,
    "order_tacos": order_tacos,
    "light_on": light_on,
    "chatbot": chat_bot,
}
