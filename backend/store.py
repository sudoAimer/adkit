"""以原子 JSON 文件持久化任务，单进程锁保护 API 与后台线程的修改。"""
import copy
import json
import re
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import HTTPException


class TaskStore:
    """管理任务记录；所有读写均返回副本，避免未加锁的共享状态修改。"""

    def __init__(self, root: Path):
        """创建数据目录，并将上次退出时未完成的作业标为失败。"""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        for file in self.root.glob("*/task.json"):
            task = json.loads(file.read_text(encoding="utf-8"))
            if task["job"]["state"] in {"queued", "running"}:
                task["job"].update(state="failed", message="服务已重启，请重新执行任务。")
                self.save(task)

    def directory(self, task_id: str) -> Path:
        """仅允许系统生成的标识符参与路径拼接。"""
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise HTTPException(404, "任务不存在")
        return self.root / task_id

    def read(self, task_id: str) -> dict:
        """读取独立任务快照。"""
        with self.lock:
            file = self.directory(task_id) / "task.json"
            if not file.is_file():
                raise HTTPException(404, "任务不存在")
            return json.loads(file.read_text(encoding="utf-8"))

    def save(self, task: dict) -> dict:
        """先写临时文件再替换，避免进程中断留下半份 JSON。"""
        with self.lock:
            directory = self.directory(task["id"])
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / f"{uuid4().hex}.tmp"
            temporary.write_text(json.dumps(task, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temporary.replace(directory / "task.json")
            return copy.deepcopy(task)

    def all(self) -> list[dict]:
        """按创建时间倒序列出任务，保证列表刷新时顺序稳定。"""
        with self.lock:
            tasks = [self.read(file.parent.name) for file in self.root.glob("*/task.json")]
            return sorted(tasks, key=lambda item: item["created_at"], reverse=True)

    def idle(self, task_id: str) -> dict:
        """在修改样本或提交作业前阻止操作正在运行的任务。"""
        task = self.read(task_id)
        if task["job"]["state"] in {"queued", "running"}:
            raise HTTPException(409, "任务正在处理中，请等待完成")
        return task
