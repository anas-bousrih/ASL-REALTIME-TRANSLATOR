from collections import deque


class TranscriptBuffer:
    def __init__(self, max_tokens=30):
        self.tokens = deque(maxlen=max_tokens)

    def add(self, token):
        if token:
            self.tokens.append(token)

    def text(self):
        return " ".join(self.tokens)

    def clear(self):
        self.tokens.clear()
