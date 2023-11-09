import threading
import time

class NotesThread(threading.Thread):
    def __init__(self, name, args=()):
        super(NotesThread, self).__init()
        self.name = name
        self.args = args

    def run(self):
        print(f"Starting {self.name}")
        # run functions
        print(f"Exiting {self.name}")