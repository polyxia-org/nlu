import openai


class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def ask(self, message):
        return self.__call__(message)

    def execute(self):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.messages
        )
        return completion.choices[0].message.content
