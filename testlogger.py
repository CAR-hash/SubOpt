import sys


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        # 强制指定 utf-8 编码打开文件
        self.log = open(filename, "w", encoding="utf-8")
        self._previous_stdout = None

    def __enter__(self):
        self._previous_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        prev = self._previous_stdout
        self._previous_stdout = None
        if prev is not None:
            sys.stdout = prev
        if self.log is not None and not self.log.closed:
            self.log.close()
        return False

    def close(self):
        """Restore ``sys.stdout`` and close the log file (legacy non-context use)."""
        if self._previous_stdout is not None:
            sys.stdout = self._previous_stdout
            self._previous_stdout = None
        elif sys.stdout is self:
            sys.stdout = self.terminal
        if self.log is not None and not self.log.closed:
            self.log.close()

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Windows consoles can be cp1252; fall back to a safe representation for terminal output.
            safe = message.encode("ascii", errors="replace").decode("ascii")
            self.terminal.write(safe)
        self.log.write(message)
        self.flush()  # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()