"""Publish native model training only after a complete candidate succeeds."""

from copy import deepcopy
from functools import wraps


def atomic_fit(fit):
    @wraps(fit)
    def train(self, data):
        from truthound.ml.base import ModelState

        with self._lock:
            candidate = deepcopy(self)
            try:
                fit(candidate, data)
            except Exception as error:
                if not self.is_trained:
                    self._state = ModelState.ERROR
                    self._error = error
                raise
            candidate._error = None
            self.__dict__.update(
                {key: value for key, value in candidate.__dict__.items() if key != "_lock"}
            )

    return train
