
from collections import deque
import random


class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory = deque()

    def add(self, trajectory):

        if self.memory_counter < self.memory_size:
            self.memory.append(trajectory)
            self.memory_counter += 1
        else:
            self.memory.popleft()
            self.memory.append(trajectory)

    def clear(self):
        self.memory.clear()
        self.memory_counter = 0

    def sample_batch(self, batch_size):
        if self.memory_counter < batch_size:
            print('insufficient memory')
            return random.sample(self.memory, self.memory_counter)
            # return False
        else:
            return random.sample(self.memory, batch_size)


