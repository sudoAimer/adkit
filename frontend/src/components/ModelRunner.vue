<script setup>
import { computed, ref, watch } from "vue";
import { request } from "../api";
const props = defineProps({ task: Object, catalog: Array, disabled: Boolean });
const emit = defineEmits(["refresh", "operation"]);
const selected = ref([...props.task.algorithms]);
const busy = ref(false),
  error = ref("");
const options = computed(() => [
  ...props.catalog,
  ...props.task.algorithms
    .filter((id) => !props.catalog.some((m) => m.id === id))
    .map((id) => ({ id, label: id, removed: true })),
]);
const names = {
  idle: "待处理",
  queued: "排队中",
  running: "运行中",
  done: "完成",
  failed: "失败",
};
const canAct = computed(() => !busy.value && !props.disabled);
watch(
  () => props.task.id,
  () => {
    selected.value = [...props.task.algorithms];
  },
);
async function act(operation, ids = selected.value) {
  if (!canAct.value || !ids.length) return;
  busy.value = true;
  error.value = "";
  try {
    await request(
      `/tasks/${props.task.id}/${operation === "add" ? "models" : `jobs/${operation}`}`,
      {
        method: operation === "add" ? "PATCH" : "POST",
        body: JSON.stringify({ algorithms: ids }),
      },
    );
    emit("refresh");
    if (operation !== "add") emit("operation", operation);
  } catch (exc) {
    error.value = exc.message;
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <section class="panel model-runner">
    <div class="section-heading">
      <div>
        <h2>选择要执行的模型</h2>
        <p>
          同一任务共享图片、尺寸与增强；勾选决定本次执行，不影响下方展示选择。
        </p>
      </div>
      <div class="button-row">
        <button
          class="text-button"
          :disabled="!canAct"
          @click="selected = catalog.map((m) => m.id)"
        >
          全选模型</button
        ><button class="text-button" :disabled="!canAct" @click="selected = []">
          清空选择
        </button>
      </div>
    </div>
    <div class="model-picker" aria-label="执行模型选择">
      <label v-for="model in options" :key="model.id" class="model-choice"
        ><input
          type="checkbox"
          :value="model.id"
          v-model="selected"
          :disabled="!canAct || model.removed"
        /><strong>{{ model.label }}</strong
        ><small>{{
          model.removed
            ? "未注册，保留历史结果"
            : !model.ready
              ? "权重未就绪"
              : "可用"
        }}</small></label
      >
    </div>
    <div class="button-row model-batch-actions">
      <button
        class="button secondary"
        :disabled="!canAct || !selected.length"
        @click="act('add')"
      >
        加入任务
      </button>
      <button
        class="button primary"
        :disabled="!canAct || !selected.length || !task.normal.length"
        @click="act('fit')"
      >
        为所选模型建库（{{ selected.length }}）
      </button>
      <button
        class="button secondary"
        :disabled="!canAct || !selected.length || !task.test.length"
        @click="act('predict')"
      >
        测试所选模型（{{ selected.length }}）
      </button>
    </div>
    <p class="threshold-note">
      可随时增选模型。新增模型首次执行时自动加入任务；测试时未建库的模型会单独报错，其他模型继续。
    </p>
    <p v-if="error" class="inline-note invalid" role="alert">{{ error }}</p>
    <div class="assessment-table-wrap">
      <table class="assessment-table" aria-label="各模型运行状态">
        <thead>
          <tr>
            <th>任务中的模型</th>
            <th>参考库</th>
            <th>当前状态</th>
            <th>运行记录</th>
            <th>单独操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="id in task.algorithms" :key="id">
            <th>{{ options.find((m) => m.id === id)?.label || id }}</th>
            <td>
              {{
                task.models[id].model_ready
                  ? `就绪 · 正常样本 v${task.models[id].fit_revision}`
                  : "待建库"
              }}
            </td>
            <td>
              <strong>{{
                names[task.models[id].state] || task.models[id].state
              }}</strong>
              <div>{{ task.models[id].message }}</div>
              <progress
                v-if="
                  task.models[id].state === 'running' && task.models[id].total
                "
                :value="task.models[id].completed"
                :max="task.models[id].total"
              />
            </td>
            <td>
              <span>{{ task.models[id].history.length }} 次运行</span>
              <div>
                {{
                  task.models[id].results_revision == null
                    ? "暂无当前结果"
                    : `结果 · 数据 v${task.models[id].results_revision}`
                }}
              </div>
              <details v-if="task.models[id].history.length">
                <summary>查看记录</summary>
                <ul class="run-history">
                  <li
                    v-for="run in [...task.models[id].history].reverse()"
                    :key="run.id"
                  >
                    {{ run.operation === "fit" ? "建库" : "测试" }} ·
                    {{ names[run.state] || run.state }} · 数据 v{{
                      run.data_revision
                    }}
                    · {{ run.started_at
                    }}<span v-if="run.error"> · {{ run.error }}</span>
                  </li>
                </ul>
              </details>
            </td>
            <td>
              <div class="button-row">
                <button
                  class="text-button"
                  :disabled="
                    !canAct ||
                    !task.normal.length ||
                    !catalog.some((m) => m.id === id)
                  "
                  @click="act('fit', [id])"
                >
                  {{
                    task.models[id].model_ready ? "重新建库" : "建库"
                  }}</button
                ><button
                  class="text-button"
                  :disabled="
                    !canAct ||
                    !task.test.length ||
                    !catalog.some((m) => m.id === id)
                  "
                  @click="act('predict', [id])"
                >
                  {{ task.models[id].state === "failed" ? "重试测试" : "测试" }}
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="threshold-note">
      当前图片版本 v{{ task.data_revision }} · 正常样本版本 v{{
        task.normal_revision
      }}。只重跑所选模型；数据变化会使依赖它的结果失效。
    </p>
  </section>
</template>
