""" optional file not beign used in the current codebase but could be used when we want to implement an event-driven architecture instead of the direct orchestration architecture we have now. 

- Article further explaining the implementation : https://oneuptime.com/blog/post/2026-02-02-python-event-driven-systems/view#:~:text=If%20you've%20ever%20built,ready%20implementations%20with%20message%20queues. 

"""


class EventBus:
    def __init__(self):
        self._handlers = {}

    def on(self, name, handler):
        if name not in self._handlers:
            self._handlers[name] = []
        self._handlers[name].append(handler)

    def emit(self, name, data=None):
        for handler in self._handlers.get(name, []):
            handler(data)



