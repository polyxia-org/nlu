import asyncio
import os

from surrealdb import Surreal


class Singleton(type):
    _instances = {}

    async def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            await instance.initialize_db()
        return cls._instances[cls]


class Database(metaclass=Singleton):
    def __init__(self) -> None:
        self.session = None

    async def initialize_db(self):
        self.session = Surreal(os.getenv("DATABASE_URL", "ws://localhost:8000/rpc"))
        await self.session.connect()
        await self.session.signin(
            {
                "user": os.getenv("DATABASE_USERNAME", "root"),
                "pass": os.getenv("DATABASE_PASSWORD", "root"),
            }
        )
        await self.session.use("polyxia", "polyxia")
