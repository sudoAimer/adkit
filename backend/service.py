"""Registry-driven model runs with isolated results and shared-data revisions."""
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from .task_models import model_state, run_snapshot


def now():
    return datetime.now(timezone.utc).isoformat()


class TaskService:
    def __init__(self, store, engine):
        self.store, self.engine = store, engine
        self.registry = engine.registry
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='adkit')

    def validate_models(self, algorithms):
        selected = list(dict.fromkeys(algorithms))
        if not selected or any(a not in self.registry for a in selected):
            raise HTTPException(400, '请选择已注册的模型')
        return selected

    def attach(self, task, algorithms):
        for algorithm in self.validate_models(algorithms):
            if algorithm not in task['models']:
                task['algorithms'].append(algorithm)
                task['models'][algorithm] = dict(model_state(algorithm),
                    parameters=dict(self.registry[algorithm].parameters))

    def create(self, values):
        algorithms = self.validate_models(values['algorithms'])
        task = dict(values, algorithms=[], models={}, id=uuid4().hex, created_at=now(),
                    schema_version=3, data_revision=0, normal_revision=0, label_revision=0,
                    thresholds={}, alignment='pad', normal=[], test=[], results=[], model_ready=False,
                    job=dict(state='idle', operation=None, message='等待上传正常图片', completed=0, total=0))
        self.attach(task, algorithms)
        return self.store.save(task)

    def add_models(self, task_id, algorithms):
        with self.store.lock:
            task = self.store.idle(task_id)
            self.attach(task, algorithms)
            return self.store.save(task)

    def clear_results(self, task, algorithms):
        retained = []
        directory = self.store.directory(task['id']) / 'results'
        for result in task['results']:
            if result['algorithm'] not in algorithms:
                retained.append(result)
                continue
            for key in ['raw_map', 'heatmap', 'overlay']:
                if result.get(key):
                    (directory / result[key].split('/')[-1]).unlink(missing_ok=True)
        task['results'] = retained
        for algorithm in algorithms:
            task['models'][algorithm]['results_revision'] = None

    def invalidate(self, task, normal_changed):
        task['data_revision'] += 1
        task['results'] = []
        directory = self.store.directory(task['id'])
        shutil.rmtree(directory / 'results', ignore_errors=True)
        if normal_changed:
            task['normal_revision'] += 1
        for state in task['models'].values():
            state.update(state='idle', results_revision=None, completed=0, total=0,
                         message='样本已更新，请重新建库' if normal_changed else '测试图片已更新，请重新测试')
            if normal_changed:
                state.update(model_ready=False, fit_revision=None)
                (directory / state['checkpoint']).unlink(missing_ok=True)
        task['model_ready'] = any(m['model_ready'] for m in task['models'].values())
        task['job'] = dict(state='idle', operation=None, message='样本已更新', completed=0, total=0)

    def submit(self, task_id, operation, algorithms):
        selected = self.validate_models(algorithms)
        with self.store.lock:
            task = self.store.idle(task_id)
            if sum(t['job']['state'] in {'queued', 'running'} for t in self.store.all()) >= 16:
                raise HTTPException(429, '等待任务较多，请稍后重试')
            if not task['normal' if operation == 'fit' else 'test']:
                raise HTTPException(400, '请先上传正常图片' if operation == 'fit' else '请先上传待测图片')
            if operation == 'analyze' and not task['normal']:
                raise HTTPException(400, '请先上传正常参考图片')
            self.attach(task, selected)
            for algorithm in selected:
                task['models'][algorithm].update(state='queued', message='排队等待处理', completed=0,
                    total=len(task['test']) if operation == 'predict' else 0)
            task['job'] = dict(state='queued', operation=operation, algorithms=selected,
                               completed=0, total=len(selected), message='所选模型排队中')
            snapshot = self.store.save(task)
            self.executor.submit(self.work, task_id, operation, selected)
            return snapshot

    def work(self, task_id, operation, algorithms):
        failures = 0
        for index, algorithm in enumerate(algorithms):
            def progress(**changes):
                with self.store.lock:
                    task = self.store.read(task_id)
                    result = changes.pop('result', None)
                    if result is not None:
                        result.update(algorithm=algorithm, data_revision=task['data_revision'])
                        task['results'].append(result)
                    task['models'][algorithm].update(changes)
                    task['job']['message'] = f"{self.registry[algorithm].label}：{changes.get('message', '处理中')}"
                    self.store.save(task)
            try:
                with self.store.lock:
                    task = self.store.read(task_id)
                    state = task['models'][algorithm]
                    task['job'].update(state='running', active_model=algorithm, completed=index)
                    record = run_snapshot(task, algorithm, operation, uuid4().hex, now())
                    state['history'].append(record)
                    self.store.save(task)
                    if operation == 'predict' and (not state['model_ready'] or state['fit_revision'] != task['normal_revision']):
                        raise ValueError('尚无当前正常样本的参考库，请先建库；其他模型继续运行')
                    self.clear_results(task, [algorithm])
                    if operation == 'fit':
                        state.update(model_ready=False, fit_revision=None)
                        (self.store.directory(task_id) / state['checkpoint']).unlink(missing_ok=True)
                    state.update(state='running', message='正在加载模型')
                    self.store.save(task)
                if operation == 'analyze':
                    if not state['model_ready'] or state['fit_revision'] != task['normal_revision']:
                        self.engine.execute(task, self.store.directory(task_id), 'fit', progress, algorithm)
                        with self.store.lock:
                            task = self.store.read(task_id)
                            task['models'][algorithm].update(model_ready=True, fit_revision=task['normal_revision'])
                            self.store.save(task)
                    self.engine.execute(task, self.store.directory(task_id), 'predict', progress, algorithm)
                else:
                    self.engine.execute(task, self.store.directory(task_id), operation, progress, algorithm)
                with self.store.lock:
                    task = self.store.read(task_id)
                    state = task['models'][algorithm]
                    failed_images = sum(bool(r.get('error')) for r in task['results'] if r['algorithm'] == algorithm)
                    failed = operation != 'fit' and failed_images == len(task['test'])
                    failures += int(failed)
                    state.update(state='failed' if failed else 'done', message='建库完成' if operation == 'fit' else f'测试完成，{failed_images} 张失败')
                    if operation == 'fit':
                        state.update(model_ready=True, fit_revision=task['normal_revision'])
                    else:
                        state['results_revision'] = task['data_revision']
                    state['history'][-1].update(state=state['state'], finished_at=now(), failed_images=failed_images)
                    self.store.save(task)
            except Exception as exc:
                failures += 1
                logging.exception('Model job failed: %s/%s', task_id, algorithm)
                with self.store.lock:
                    task = self.store.read(task_id)
                    state = task['models'][algorithm]
                    state.update(state='failed', message=f'处理失败：{exc}')
                    if state['history']:
                        state['history'][-1].update(state='failed', error=str(exc), finished_at=now())
                    self.store.save(task)
        with self.store.lock:
            task = self.store.read(task_id)
            task['job'].update(state='failed' if failures == len(algorithms) else 'done',
                               completed=len(algorithms), active_model=None,
                               message=f'所选 {len(algorithms)} 个模型处理结束，{failures} 个失败', finished_at=now())
            self.store.save(task)

    def close(self):
        self.executor.shutdown(wait=True)
