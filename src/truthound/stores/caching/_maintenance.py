"""Instance-owned cache refresh loop; source I/O remains in CachedStore."""

from threading import Event, Thread


class Maintenance:
    def __init__(self, owner):
        self.owner = owner
        self.stopped = Event()
        self.thread = None
        self.failures = 0
        self.refreshes = 0

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        config = self.owner.cache.config
        if not config.background_refresh or config.ttl_seconds == 0:
            return
        self.owner.cache.expiring_keys(config.refresh_threshold_percent)
        self.stopped.clear()
        self.thread = Thread(target=self.run, name="truthound-cache-refresh", daemon=True)
        self.thread.start()

    def run(self):
        config = self.owner.cache.config
        interval = max(
            0.01, min(60.0, config.ttl_seconds * max(config.refresh_threshold_percent, 1) / 200)
        )
        while not self.stopped.wait(interval):
            try:
                for key in self.owner.cache.expiring_keys(config.refresh_threshold_percent):
                    if self.stopped.is_set():
                        break
                    if key.startswith("store:"):
                        self.owner.refresh(key[6:])
                        self.refreshes += 1
            except Exception:
                self.failures += 1

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def stats(self):
        return {
            "running": self.thread is not None and self.thread.is_alive(),
            "failures": self.failures,
            "refreshes": self.refreshes,
        }
