# utils/timer.py
"""Simple timer for performance tracking."""

import time
from contextlib import contextmanager


class Timer:
    """Performance timer with section tracking."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """Stop timing a section."""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0.0
    
    @contextmanager
    def time_section(self, name: str):
        """Context manager for timing a code block."""
        self.start(name)
        yield
        self.stop(name)
    
    def get_times(self) -> dict:
        """Get all recorded times."""
        return self.times.copy()
