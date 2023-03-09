
import time
import numpy as np
from contextlib import contextmanager


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.times = []

    def recent_average_time(self, latest_n):
        return np.mean(np.array(self.times)[-latest_n:])

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.times.append(self.diff)
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    @contextmanager
    def timed(self):
        self.tic()
        yield
        self.toc()


class Timers(object):
    def __init__(self):
        self._timers = {}

    def tic(self, key):
        if key not in self._timers:
            self._timers[key] = Timer()
        self._timers[key].tic()

    def toc(self, key):
        self._timers[key].toc()

    @contextmanager
    def timed(self, key):
        self.tic(key)
        yield
        self.toc(key)

    def __str__(self):
        msg = []
        for k, v in self._timers.items():
            msg.append('%s: %f' % (k, v.average_time))
        return ', '.join(msg)