"""
AI 聊天面板
职责：显示日志和 AI 对话，管理用户输入
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
from typing import Optional


class ChatPanel:
    """AI 聊天 / 日志面板"""

    def __init__(self, parent: tk.Widget):
        self.parent = parent

        # 聊天记录
        self.frame = ttk.Labelframe(parent, text="AI Assistant Chat")
        self.frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.history = scrolledtext.ScrolledText(
            self.frame, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.history.insert(tk.END, "System: Welcome to LabGuardian.\n")
        self.history.config(state=tk.DISABLED)

        # 用户输入框
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(input_frame, text="Ask:").pack(side=tk.LEFT)
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def log(self, text: str):
        """添加一条日志到聊天面板"""
        self.history.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.history.insert(tk.END, f"[{timestamp}] {text}\n")
        self.history.see(tk.END)
        self.history.config(state=tk.DISABLED)

    def get_user_input(self) -> str:
        """获取并清空用户输入"""
        text = self.input_entry.get().strip()
        self.input_entry.delete(0, tk.END)
        return text

    def bind_send(self, callback):
        """绑定回车发送事件"""
        self.input_entry.bind("<Return>", lambda e: callback())
