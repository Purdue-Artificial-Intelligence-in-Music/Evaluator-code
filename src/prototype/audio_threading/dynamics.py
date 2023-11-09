import threading
import time

class DynamicsThread(threading.Thread):
    def __init__(self, name, args=()):
        super(DynamicsThread, self).__init()
        self.name = name
        self.args = args

    def run(self):
        print(f"Starting {self.name}")
        # run functions
        print(f"Exiting {self.name}")