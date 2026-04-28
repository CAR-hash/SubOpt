import sys


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        # 强制指定 utf-8 编码打开文件
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()