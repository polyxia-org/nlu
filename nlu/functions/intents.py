from nlu.functions import (
    chat_bot,
    get_date,
    get_weather,
    light_on,
    order_tacos,
    say_hello,
)

INTENTS_HANDLER = {
    "GetDate": get_date,
    "GetWeather": get_weather,
    "SayHello": say_hello,
    "OrderTacos": order_tacos,
    "LightOn": light_on,
    "Chatbot": chat_bot,
}
