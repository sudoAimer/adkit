"""Versioned task layout and one-time migration of historical task records."""
import copy


def model_state(algorithm):
    return dict(state='idle', message='待建库', model_ready=False,
                checkpoint=f'model_{algorithm}.pt', fit_revision=None,
                results_revision=None, completed=0, total=0, history=[])


def normalize(task):
    if 'algorithms' not in task:
        old = task.pop('algorithm', 'anomalydino')
        # Compatibility only: new tasks and execution never use comparison as a model.
        task['algorithms'] = ['anomalydino', 'subspacead'] if old == 'comparison' else [old]
        task['data_revision'] = 0
        task['normal_revision'] = 0
        task['models'] = {}
        for algorithm in task['algorithms']:
            state = model_state(algorithm)
            if old != 'comparison':
                state['checkpoint'] = 'model.pt'
            state['model_ready'] = task.get('model_ready', False)
            state['fit_revision'] = 0 if state['model_ready'] else None
            state['state'] = 'done' if state['model_ready'] else 'idle'
            state['message'] = '参考库已就绪' if state['model_ready'] else '待建库'
            for result in task.get('results', []):
                result.setdefault('algorithm', old)
                result.setdefault('data_revision', 0)
                if result['algorithm'] == algorithm:
                    state['results_revision'] = 0
            task['models'][algorithm] = state
        if old != 'comparison':
            task.setdefault('thresholds', {}).setdefault(old, {
                'threshold': task.get('threshold'), 'area_threshold': task.get('area_threshold', 0)})
    task.setdefault('label_revision', 0)
    task['schema_version'] = 3
    task['model_ready'] = any(m['model_ready'] for m in task['models'].values())
    return task


def run_snapshot(task, algorithm, operation, run_id, started_at):
    return dict(id=run_id, operation=operation, state='running', started_at=started_at,
                data_revision=task['data_revision'], normal_revision=task['normal_revision'],
                label_revision=task['label_revision'],
                normal_ids=[i['id'] for i in task['normal']],
                test_ids=[i['id'] for i in task['test']],
                image_size=copy.deepcopy(task['image_size']), rotation=task['rotation'],
                alignment=task.get('alignment', 'legacy'),
                parameters=copy.deepcopy(task['models'][algorithm].get('parameters', {})))
