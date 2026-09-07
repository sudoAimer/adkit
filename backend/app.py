"""FastAPI 路由与静态前端托管；通过工厂创建单进程应用。"""
import os
import shutil
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .engine import DetectorEngine
from .schemas import TaskCreate, ThresholdUpdate, LabelUpdate, ModelSelection
from .service import TaskService
from .store import TaskStore
from .assessment import assess, task_algorithms
from .registry import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 12_000_000


def create_app() -> FastAPI:
    """读取服务器环境变量，模型权重仅在实际作业启动时加载。"""
    store = TaskStore(Path(os.getenv("ADKIT_DATA_DIR", str(ROOT / "web-data"))))
    registry = dict(MODEL_REGISTRY)
    weights = {id: spec.weights(ROOT) for id, spec in registry.items()}
    engine = DetectorEngine(weights, os.getenv("ADKIT_DEVICE", "cuda"), registry)
    service = TaskService(store, engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """关闭服务前释放作业线程，保留磁盘上的任务和参考库。"""
        yield
        service.close()

    app = FastAPI(title="adkit 图像检测工作台", lifespan=lifespan)

    @app.get("/api/settings")
    def settings():
        """返回算法资源准备状态，不向网页暴露服务器绝对路径。"""
        return {"algorithms": [{"id": id, "label": spec.label, "ready": spec.available(weights[id])}
                                for id, spec in registry.items()]}

    @app.get("/api/tasks")
    def list_tasks():
        """返回单人工作区的全部任务及最新状态。"""
        return store.all()

    @app.post("/api/tasks", status_code=201)
    def create_task(body: TaskCreate):
        """创建可持续补充样本的检测任务。"""
        return service.create(body.model_dump())

    @app.patch('/api/tasks/{task_id}/models')
    def add_models(task_id: str, body: ModelSelection):
        return service.add_models(task_id, body.algorithms)

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        """提供页面刷新与任务轮询所需的完整快照。"""
        return store.read(task_id)

    @app.delete("/api/tasks/{task_id}", status_code=204)
    def delete_task(task_id: str):
        """只允许删除空闲任务，同时删除其图片、参考库和结果。"""
        with store.lock:
            store.idle(task_id)
            shutil.rmtree(store.directory(task_id))

    @app.patch("/api/tasks/{task_id}/threshold")
    def threshold(task_id: str, body: ThresholdUpdate):
        """仅保存图像判定阈值，已有分数与热力图无需重新计算。"""
        with store.lock:
            task = store.read(task_id)
            algorithm = body.algorithm or (task['algorithms'][0] if len(task['algorithms']) == 1 else None)
            if algorithm not in task_algorithms(task):
                raise HTTPException(400, '请选择此任务中的模型')
            task.setdefault('thresholds', {})[algorithm] = {
                'threshold': body.threshold, 'area_threshold': body.area_threshold}
            return store.save(task)

    @app.post('/api/tasks/{task_id}/assessment')
    def assessment(task_id: str, body: ThresholdUpdate):
        with store.lock:
            task = store.read(task_id)
            algorithm = body.algorithm or (task['algorithms'][0] if len(task['algorithms']) == 1 else None)
            if algorithm not in task_algorithms(task):
                raise HTTPException(400, '请选择此任务中的模型')
            return assess(task, store.directory(task_id), algorithm, body.threshold, body.area_threshold)

    @app.patch('/api/tasks/{task_id}/images/test/{image_id}/label')
    def label_image(task_id: str, image_id: str, body: LabelUpdate):
        with store.lock:
            task = store.read(task_id)
            item = next((i for i in task['test'] if i['id'] == image_id), None)
            if item is None:
                raise HTTPException(404, '图片不存在')
            item['label'] = body.label
            task['label_revision'] += 1
            return store.save(task)

    @app.post("/api/tasks/{task_id}/images/{kind}", status_code=201)
    def upload(task_id: str, kind: Literal["normal", "test"], files: list[UploadFile] = File(...)):
        """校验并转存图片；整批成功后才更新任务，失败则清理本批文件。"""
        created = []
        try:
            if not 1 <= len(files) <= 20:
                raise HTTPException(400, "每次请选择 1–20 张图片")
            with store.lock:
                task = store.idle(task_id)
                if len(task[kind]) + len(files) > 200:
                    raise HTTPException(400, "每组最多保存 200 张图片")
                directory = store.directory(task_id) / "images"
                directory.mkdir(exist_ok=True)
                items = []
                for upload_file in files:
                    content = upload_file.file.read(MAX_BYTES + 1)
                    if len(content) > MAX_BYTES:
                        raise HTTPException(413, "单张图片不能超过 20 MB")
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", Image.DecompressionBombWarning)
                            with Image.open(BytesIO(content)) as image:
                                if image.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                                    raise ValueError("不支持的图片格式")
                                w, h = image.size
                                if w * h > MAX_PIXELS:
                                    raise ValueError("图片不能超过 1200 万像素")
                                rgb = ImageOps.exif_transpose(image).convert("RGB")
                                item_id = uuid4().hex
                                filename = f"{item_id}.png"
                                destination = directory / filename
                                created.append(destination)
                                rgb.save(destination)
                    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
                        raise HTTPException(400, f"图片无效：{upload_file.filename}（{exc}）") from exc
                    items.append({"id": item_id, "name": (upload_file.filename or "图片")[:200], "file": filename,
                                  "url": f"/api/tasks/{task_id}/files/images/{filename}", "label": None})
                task[kind].extend(items)
                service.invalidate(task, normal_changed=kind == "normal")
                return store.save(task)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        finally:
            for upload_file in files:
                upload_file.file.close()

    @app.delete("/api/tasks/{task_id}/images/{kind}/{image_id}")
    def remove_image(task_id: str, kind: Literal["normal", "test"], image_id: str):
        """删除指定样本，并使依赖该样本的参考状态或结果失效。"""
        with store.lock:
            task = store.idle(task_id)
            item = next((item for item in task[kind] if item["id"] == image_id), None)
            if item is None:
                raise HTTPException(404, "图片不存在")
            task[kind].remove(item)
            service.invalidate(task, normal_changed=kind == "normal")
            saved = store.save(task)
            (store.directory(task_id) / "images" / item["file"]).unlink(missing_ok=True)
            return saved

    @app.post("/api/tasks/{task_id}/jobs/{operation}", status_code=202)
    def submit(task_id: str, operation: Literal["fit", "predict"], body: ModelSelection):
        """提交串行后台作业，立即返回排队状态供页面轮询。"""
        return service.submit(task_id, operation, body.algorithms)

    @app.get("/api/tasks/{task_id}/files/{folder}/{filename}")
    def image_file(task_id: str, folder: Literal["images", "results"], filename: str):
        """仅提供任务记录中登记的图片，禁止访问检查点或任意本地文件。"""
        task = store.read(task_id)
        allowed = {item["file"] for item in task["normal"] + task["test"]} if folder == "images" else {
            Path(item[key]).name for item in task["results"] for key in ("heatmap", "overlay") if key in item
        }
        if filename not in allowed:
            raise HTTPException(404, "图片不存在")
        path = store.directory(task_id) / folder / filename
        if not path.is_file():
            raise HTTPException(404, "图片不存在")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})

    # 构建后的 Vue 前端与 API 同源，部署时无需额外配置跨域。
    frontend = Path(os.getenv("ADKIT_FRONTEND_DIST", str(ROOT / "frontend/dist")))
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
