<script setup>
import { computed, onUnmounted, reactive, ref, watch } from "vue";
import { request } from "../api";
const props = defineProps({ task: { type: Object, required: true } });
const emit = defineEmits(["refresh"]);
const models = computed(() =>
  props.task.algorithm === "comparison"
    ? ["anomalydino", "subspacead"]
    : [props.task.algorithm],
);
const names = { anomalydino: "AnomalyDINO", subspacead: "SubspaceAD" };
const active = ref(models.value[0]);
const drafts = reactive(
  Object.fromEntries(
    models.value.map((model) => {
      const saved = props.task.thresholds?.[model] || props.task;
      return [
        model,
        {
          threshold: saved.threshold ?? "",
          area_threshold: saved.area_threshold ?? 0,
        },
      ];
    }),
  ),
);
const reports = ref({}),
  error = ref(""),
  notice = ref(""),
  updating = ref(false),
  saving = ref(false);
const selectedId = ref(""),
  view = ref("overlay"),
  filter = ref("all");
let timeout,
  controller,
  sequence = 0;
const draft = computed(() => drafts[active.value]);
const successful = computed(() =>
  props.task.results.filter((r) => Number.isFinite(r.score)),
);
const scoreMax = computed(() =>
  Math.max(
    0.1,
    ...successful.value
      .filter((r) => (r.algorithm || props.task.algorithm) === active.value)
      .map((r) => r.score * 1.25),
    props.task.thresholds?.[active.value]?.threshold ||
      props.task.threshold ||
      0,
  ),
);
const areaMax = computed(() =>
  Math.max(
    1,
    ...successful.value.map((r) => r.pixels || 0),
    props.task.thresholds?.[active.value]?.area_threshold ||
      props.task.area_threshold ||
      0,
  ),
);
const valid = (model) => {
  const d = drafts[model];
  return (
    (d.threshold === "" ||
      (Number.isFinite(Number(d.threshold)) && Number(d.threshold) >= 0)) &&
    d.area_threshold !== "" &&
    Number.isSafeInteger(Number(d.area_threshold)) &&
    Number(d.area_threshold) >= 0
  );
};
const payload = (model) => ({
  algorithm: model,
  threshold:
    drafts[model].threshold === "" ? null : Number(drafts[model].threshold),
  area_threshold: Number(drafts[model].area_threshold),
});
const changed = computed(() => {
  const saved = props.task.thresholds?.[active.value] || props.task;
  return (
    payload(active.value).threshold !== (saved.threshold ?? null) ||
    Number(draft.value.area_threshold) !== (saved.area_threshold ?? 0)
  );
});
function resultFor(id, model) {
  return props.task.results.find(
    (r) => r.id === id && (r.algorithm || props.task.algorithm) === model,
  );
}
function decision(id, model) {
  return updating.value
    ? "unset"
    : reports.value[model]?.rows.find((r) => r.id === id)?.classification ||
        "unset";
}
const labels = {
  defect: "缺陷",
  normal: "正常",
  error: "检测失败",
  unset: "未判定",
};
const visible = computed(() =>
  props.task.test.filter(
    (item) =>
      filter.value === "all" ||
      decision(item.id, active.value) === filter.value,
  ),
);
const selected = computed(
  () =>
    visible.value.find((i) => i.id === selectedId.value) || visible.value[0],
);
const text = (n) => (Number.isFinite(n) ? Number(n).toPrecision(5) : "—");
const percent = (n) => (Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—");
const count = (model, key) =>
  updating.value || !reports.value[model]
    ? "—"
    : reports.value[model].counts[key];
function schedule() {
  clearTimeout(timeout);
  controller?.abort();
  const version = ++sequence;
  updating.value = true;
  notice.value = "";
  timeout = setTimeout(async () => {
    const abort = new AbortController();
    controller = abort;
    try {
      const entries = await Promise.all(
        models.value.map(async (model) => [
          model,
          valid(model)
            ? await request(`/tasks/${props.task.id}/assessment`, {
                method: "POST",
                body: JSON.stringify(payload(model)),
                signal: abort.signal,
              })
            : null,
        ]),
      );
      if (version !== sequence) return;
      reports.value = Object.fromEntries(entries);
      error.value = "";
    } catch (exc) {
      if (version === sequence) {
        reports.value = {};
        error.value = exc.message;
      }
    } finally {
      if (version === sequence) updating.value = false;
    }
  }, 150);
}
watch(
  () =>
    JSON.stringify([
      props.task.results,
      props.task.test.map((i) => [i.id, i.label]),
      drafts,
    ]),
  schedule,
  { immediate: true },
);
onUnmounted(() => {
  ++sequence;
  clearTimeout(timeout);
  controller?.abort();
});
async function save() {
  if (!valid(active.value) || saving.value) return;
  saving.value = true;
  try {
    await request(`/tasks/${props.task.id}/threshold`, {
      method: "PATCH",
      body: JSON.stringify(payload(active.value)),
    });
    emit("refresh");
    notice.value = "阈值已保存";
  } catch (exc) {
    error.value = exc.message;
  } finally {
    saving.value = false;
  }
}
async function setLabel(value) {
  if (!selected.value || saving.value) return;
  saving.value = true;
  try {
    await request(
      `/tasks/${props.task.id}/images/test/${selected.value.id}/label`,
      { method: "PATCH", body: JSON.stringify({ label: value || null }) },
    );
    emit("refresh");
  } catch (exc) {
    error.value = exc.message;
  } finally {
    saving.value = false;
  }
}
</script>
<template>
  <section class="panel results-panel">
    <div class="section-heading">
      <div>
        <h2>
          {{ models.length > 1 ? "同批数据 · 模型对比" : "缺陷检出结果" }}
        </h2>
        <p>
          误报、漏检与检出按已标注图片统计；两个模型共享正常样本、待测图片与标注。
        </p>
      </div>
      <span class="quiet-tag">{{ task.test.length }} 张待测图片</span>
    </div>
    <div class="assessment-table-wrap">
      <table
        class="assessment-table"
        aria-label="当前阈值下的检出统计"
        :aria-busy="updating"
      >
        <thead>
          <tr>
            <th>模型</th>
            <th>尺寸阈值（px²）</th>
            <th>分数阈值</th>
            <th>误报 FP</th>
            <th>漏检 FN</th>
            <th>检出 TP</th>
            <th>正确正常 TN</th>
            <th>未标注</th>
            <th>未判定／失败／待测</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="model in models"
            :key="model"
            :class="{ 'active-row': model === active }"
          >
            <th>{{ names[model] }}</th>
            <td>{{ drafts[model].area_threshold }}</td>
            <td>
              {{
                drafts[model].threshold === ""
                  ? "未设置"
                  : text(Number(drafts[model].threshold))
              }}
            </td>
            <td>{{ count(model, "false_positive") }}</td>
            <td>{{ count(model, "false_negative") }}</td>
            <td>{{ count(model, "detected") }}</td>
            <td>{{ count(model, "true_negative") }}</td>
            <td>{{ count(model, "unlabelled") }}</td>
            <td>
              {{ count(model, "unset") }} / {{ count(model, "failed") }} /
              {{ count(model, "pending") }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="threshold-editor">
      <div
        class="view-tabs"
        v-if="models.length > 1"
        aria-label="选择要调整阈值的模型"
      >
        <button
          v-for="model in models"
          :key="model"
          :class="{ active: active === model }"
          @click="active = model"
        >
          {{ names[model] }} 阈值
        </button>
      </div>
      <label class="threshold-slider-row"
        ><span
          ><strong>尺寸阈值</strong
          ><small>最小连通缺陷面积 · 原图像素（px²）</small></span
        ><input
          type="range"
          min="0"
          :max="areaMax"
          step="1"
          v-model.number="draft.area_threshold"
          aria-label="拖动调整尺寸阈值" /><input
          type="number"
          min="0"
          step="1"
          v-model="draft.area_threshold"
          aria-label="尺寸阈值像素面积"
      /></label>
      <label class="threshold-slider-row"
        ><span
          ><strong>分数阈值</strong><small>原始异常分数，非概率</small></span
        ><input
          type="range"
          min="0"
          :max="scoreMax"
          :step="scoreMax / 2000"
          :value="draft.threshold || 0"
          @input="draft.threshold = $event.target.value"
          aria-label="拖动调整分数阈值" /><input
          type="number"
          min="0"
          step="any"
          v-model="draft.threshold"
          placeholder="未设置"
          aria-label="分数阈值"
      /></label>
      <div class="threshold-actions">
        <span>{{
          updating
            ? "正在更新统计…"
            : changed
              ? "当前为阈值预览"
              : "使用已保存阈值"
        }}</span
        ><button
          class="button secondary"
          :disabled="saving || !valid(active) || !changed"
          @click="save"
        >
          保存 {{ names[active] }} 阈值</button
        ><button
          class="text-button"
          @click="
            draft.threshold = '';
            draft.area_threshold = 0;
          "
        >
          清空阈值
        </button>
      </div>
      <p class="threshold-note">
        尺寸为 0 时仅按图像分数判定。尺寸大于 0
        时，图像分数须达到分数阈值，且异常图中达到该分数阈值的最大 8
        连通区域须达到面积阈值。不同模型的分数尺度不同，请分别校准。
      </p>
      <p v-if="!valid(active)" class="inline-note invalid">
        分数须为非负有限数，尺寸须为非负整数。
      </p>
      <p v-if="error" class="inline-note invalid" role="alert">{{ error }}</p>
      <p v-if="notice" class="inline-note" role="status">{{ notice }}</p>
      <p v-if="!updating && reports[active]" class="threshold-note">
        误报率 {{ percent(reports[active].false_positive_rate) }} · 漏检率
        {{ percent(reports[active].miss_rate) }} · 检出率
        {{
          percent(reports[active].recall)
        }}。分母为相应已标注且已判定的正常／缺陷图片；无样本显示 —。
      </p>
    </div>
    <div class="results-toolbar">
      <div class="filter-tabs">
        <button
          v-for="option in [
            { id: 'all', name: '全部' },
            { id: 'defect', name: '缺陷' },
            { id: 'normal', name: '正常' },
          ]"
          :key="option.id"
          :class="{ active: filter === option.id }"
          @click="filter = option.id"
        >
          {{ option.name }}
        </button>
      </div>
      <span>按 {{ names[active] }} 当前阈值筛选</span>
    </div>
    <div v-if="!task.test.length" class="result-empty">
      <h3>等待上传待测图片</h3>
      <p>检测后可对比同一图片的分数与异常分布。</p>
    </div>
    <div v-else class="result-workspace comparison-workspace">
      <div class="result-list">
        <button
          v-for="item in visible"
          :key="item.id"
          :class="{ active: selected?.id === item.id }"
          @click="selectedId = item.id"
        >
          <img :src="item.url" :alt="item.name" loading="lazy" /><span
            ><strong>{{ item.name }}</strong
            ><small
              >实际：{{
                item.label === "normal"
                  ? "正常"
                  : item.label === "defect"
                    ? "缺陷"
                    : "未标注"
              }}</small
            ></span
          ><em class="badge" :class="decision(item.id, active)">{{
            labels[decision(item.id, active)]
          }}</em>
        </button>
        <p v-if="!visible.length" class="nav-empty">当前筛选下没有图片</p>
      </div>
      <div v-if="selected" class="result-detail">
        <div class="detail-heading">
          <strong>{{ selected.name }}</strong
          ><label class="truth-label"
            >实际标注<select
              :value="selected.label || ''"
              :disabled="saving"
              @change="setLabel($event.target.value)"
            >
              <option value="">未标注</option>
              <option value="normal">正常</option>
              <option value="defect">缺陷</option>
            </select></label
          >
        </div>
        <div class="view-tabs" aria-label="图片视图">
          <button
            v-for="option in [
              { id: 'original', name: '原图' },
              { id: 'heatmap', name: '热力图' },
              { id: 'overlay', name: '叠加图' },
            ]"
            :key="option.id"
            :class="{ active: view === option.id }"
            @click="view = option.id"
          >
            {{ option.name }}
          </button>
        </div>
        <div class="model-image-grid" :class="{ paired: models.length > 1 }">
          <article v-for="model in models" :key="model">
            <div class="detail-heading">
              <strong>{{ names[model] }}</strong
              ><span class="badge" :class="decision(selected.id, model)">{{
                labels[decision(selected.id, model)]
              }}</span>
            </div>
            <div class="image-viewer">
              <img
                :src="
                  view === 'original'
                    ? selected.url
                    : resultFor(selected.id, model)?.[view] || selected.url
                "
                :alt="`${selected.name} · ${names[model]} · ${view}`"
              />
            </div>
            <p
              v-if="resultFor(selected.id, model)?.error"
              class="inline-note invalid"
            >
              {{ resultFor(selected.id, model).error }}
            </p>
            <p v-else-if="!resultFor(selected.id, model)" class="inline-note">
              等待该模型检测，当前显示原图。
            </p>
            <div class="detail-footer">
              <span
                >分数
                <strong>{{
                  text(resultFor(selected.id, model)?.score)
                }}</strong></span
              ><a
                v-if="
                  resultFor(selected.id, model)?.[view] || view === 'original'
                "
                :href="
                  view === 'original'
                    ? selected.url
                    : resultFor(selected.id, model)?.[view]
                "
                download
                class="text-button"
                >下载图片</a
              >
            </div>
            <p
              v-if="
                reports[model]?.rows.find((r) => r.id === selected.id)?.reason
              "
              class="inline-note"
            >
              {{ reports[model].rows.find((r) => r.id === selected.id).reason }}
            </p>
          </article>
        </div>
        <p class="color-note">
          热力图按单张图片拉伸显示，颜色不可跨模型比较；判定使用原始分数。标注同步用于两个模型的统计，调整阈值无需重跑模型。
        </p>
      </div>
    </div>
  </section>
</template>
