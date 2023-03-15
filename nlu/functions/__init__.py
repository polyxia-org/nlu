"""Temporary folder for POC to test functions calling without using morty serverless functions"""
from nlu.functions.functions_mockup import (
    chat_bot,
    get_date,
    get_weather,
    light_on,
    order_tacos,
    say_hello,
)
from nlu.functions.intents import INTENTS_HANDLER
