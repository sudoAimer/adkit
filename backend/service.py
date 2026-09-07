"""任务生命周期与单线程作业队列，统一处理状态更新和样本失效规则。"""
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from .store import TaskStore


class TaskService:
    """单人服务使用一个作业线程，所有模型作业串行执行。"""

    def __init__(self, store: TaskStore, engine):
        """初始化受限队列，防止多个建库作业同时占用显存。"""
        self.store = store
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adkit")

    def create(self, values: dict) -> dict:
        """创建空任务，阈值初始为空，避免将未经校准的值作为判定依据。"""
        task = dict(values, id=uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
                    threshold=None, normal=[], test=[], results=[], model_ready=False,
                    job={"state": "idle", "operation": None, "message": "等待上传正常图片", "completed": 0, "total": 0})
        return self.store.save(task)

    def invalidate(self, task: dict, normal_changed: bool) -> None:
        """样本变化后清除旧结果；正常样本变化还必须重新建立参考库。"""
        directory = self.store.directory(task["id"])
        task["results"] = []
        shutil.rmtree(directory / "results", ignore_errors=True)
        if normal_changed:
            task["model_ready"] = False
            (directory / "model.pt").unlink(missing_ok=True)
        task["job"] = {"state": "idle", "operation": None, "message": "样本已更新", "completed": 0, "total": 0}

    def submit(self, task_id: str, operation: str) -> dict:
        """校验作业条件并持久化排队状态，返回后由后台线程继续处理。"""
        with self.store.lock:
            task = self.store.idle(task_id)
            if sum(t["job"]["state"] in {"queued", "running"} for t in self.store.all()) >= 16:
                raise HTTPException(429, "等待任务较多，请稍后重试")
            if operation == "fit" and not task["normal"]:
                raise HTTPException(400, "请先上传正常图片")
            if operation == "predict" and (not task["model_ready"] or not task["test"]):
                raise HTTPException(400, "请先完成建库并上传待测图片")
            self.invalidate(task, normal_changed=operation == "fit")
            task["job"] = {"state": "queued", "operation": operation, "message": "排队等待处理…",
                           "completed": 0, "total": len(task["test"]) if operation == "predict" else 0}
            snapshot = self.store.save(task)
            self.executor.submit(self.work, task_id, operation)
            return snapshot

    def work(self, task_id: str, operation: str) -> None:
        """捕获后台异常并落盘，刷新页面后仍能看到任务进度与失败原因。"""
        def progress(**changes):
            """合并作业进度与单图结果，不覆盖同时调整的阈值。"""
            with self.store.lock:
                current = self.store.read(task_id)
                result = changes.pop("result", None)
                if result is not None:
                    current["results"].append(result)
                current["job"].update(changes)
                self.store.save(current)

        try:
            progress(state="running", started_at=datetime.now(timezone.utc).isoformat())
            task = self.store.read(task_id)
            updates = self.engine.execute(task, self.store.directory(task_id), operation, progress)
            with self.store.lock:
                task = self.store.read(task_id)
                task.update(updates)
                failed = sum("error" in item for item in task["results"])
                task["job"].update(state="done", message="建库完成，可以开始检测" if operation == "fit" else f"检测完成，{failed} 张失败",
                                   finished_at=datetime.now(timezone.utc).isoformat())
                self.store.save(task)
        except Exception as exc:
            logging.exception("adkit job failed: %s", task_id)
            progress(state="failed", message=f"处理失败：{exc}", finished_at=datetime.now(timezone.utc).isoformat())

    def close(self) -> None:
        """关闭服务时等待已提交作业完成，避免中途丢失检查点。"""
        self.executor.shutdown(wait=True)
