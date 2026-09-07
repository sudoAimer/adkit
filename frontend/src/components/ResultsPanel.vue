<script setup>
import { computed, onUnmounted, reactive, ref, watch } from "vue";
import { request } from "../api";
const props = defineProps({
  task: { type: Object, required: true },
  catalog: { type: Array, default: () => [] },
  mode: { type: String, default: 'adjust' },
});
const emit = defineEmits(["refresh"]);
const allModels = computed(() => props.task.algorithms);
const shown = ref([...allModels.value]);
const models = computed(() =>
  allModels.value.filter((id) => shown.value.includes(id)),
);
const names = computed(() =>
  Object.fromEntries(
    allModels.value.map((id) => [
      id,
      props.catalog.find((m) => m.id === id)?.label || id,
    ]),
  ),
);
const active = ref(models.value[0]);
const imageModels = computed(() => props.mode === 'compare' ? models.value : [active.value].filter(Boolean));
const drafts = reactive({});
watch(
  allModels,
  (ids) => {
    for (const id of ids) if (!drafts[id] && !shown.value.includes(id)) shown.value.push(id);
    for (const id of ids) {
      if (!drafts[id]) {
        const saved = props.task.thresholds?.[id] || {};
        drafts[id] = {
          threshold: saved.threshold ?? "",
          area_threshold: saved.area_threshold ?? 0,
        };
      }
    }
  },
  { immediate: true },
);
watch(models, (ids) => {
  if (!ids.includes(active.value)) active.value = ids[0];
});
const reports = ref({}),
  error = ref(""),
  notice = ref(""),
  updating = ref(false),
  saving = ref(false);
const selectedId = ref(""),
  view = ref("overlay"),
  filter = ref("all");
watch(() => props.mode, () => { filter.value = 'all'; });
let timeout,
  controller,
  sequence = 0;
const draft = computed(
  () => drafts[active.value] || { threshold: "", area_threshold: 0 },
);
const successful = computed(() =>
  props.task.results.filter(
    (r) =>
      Number.isFinite(r.score) && r.data_revision === props.task.data_revision,
  ),
);
const scoreMax = computed(() =>
  Math.max(
    0.1,
    ...successful.value
      .filter((r) => r.algorithm === active.value)
      .map((r) => r.score * 1.25),
    props.task.thresholds?.[active.value]?.threshold ||
      props.task.threshold ||
      0,
  ),
);
const valid = (model) => {
  const d = drafts[model];
  if (!d) return false;
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
  if (!active.value) return false;
  const saved = props.task.thresholds?.[active.value] || {};
  return (
    payload(active.value).threshold !== (saved.threshold ?? null) ||
    Number(draft.value.area_threshold) !== (saved.area_threshold ?? 0)
  );
});
function resultFor(id, model) {
  return props.task.results.find(
    (r) =>
      r.id === id &&
      r.algorithm === model &&
      r.data_revision === props.task.data_revision,
  );
}
function decision(id, model) {
  return updating.value
    ? "unset"
    : reports.value[model]?.rows.find((r) => r.id === id)?.classification ||
        "unset";
}
function imageSource(model) {
  if (!selected.value) return '';
  if (view.value === 'original') return selected.value.url;
  if (['binary', 'filtered'].includes(view.value)) {
    if (!valid(model) || drafts[model].threshold === '' || !resultFor(selected.value.id, model)?.raw_map) return '';
    const query = new URLSearchParams({ threshold: drafts[model].threshold,
      area_threshold: view.value === 'filtered' ? drafts[model].area_threshold : 0,
      revision: props.task.data_revision });
    return `/api/tasks/${props.task.id}/binary/${model}/${selected.value.id}?${query}`;
  }
  return resultFor(selected.value.id, model)?.[view.value] || '';
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
      (filter.value === 'different' && new Set(models.value.map(model => decision(item.id, model)).filter(value => ['normal', 'defect'].includes(value))).size > 1) ||
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
      models.value,
      props.task.data_revision,
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
          {{ mode === 'compare' ? '对比结果' : '调整判定标准' }}
        </h2>
        <p>
          误报、漏检与检出按已标注图片统计；所有模型共享正常样本、待测图片与标注。
        </p>
      </div>
      <span class="quiet-tag">{{ task.test.length }} 张待测图片</span>
    </div>
    <div v-if="mode === 'compare'" class="display-models">
      <strong>参与对比的算法</strong>
      <div class="model-picker" aria-label="展示模型选择">
        <label v-for="id in allModels" :key="id" class="model-choice"
          ><input type="checkbox" :value="id" v-model="shown" />{{
            names[id]
          }}</label
        >
      </div>
      <div class="button-row">
        <button class="text-button" @click="shown = [...allModels]">
          展示全部</button
        ><button class="text-button" @click="shown = []">清空展示</button>
      </div>
    </div>
    <p v-if="!models.length" class="inline-note">请选择要展示的模型。</p>
    <template v-if="models.length">
      <div class="assessment-table-wrap">
        <table
          class="assessment-table"
          aria-label="当前阈值下的检出统计"
          :aria-busy="updating"
        >
          <thead>
            <tr>
              <th>模型</th>
              <th>最小异常面积（px²）</th>
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
              @click="active = model"
            >
              <th>
                <button class="text-button" @click="active = model">
                  {{ names[model] }}
                </button>
              </th>
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
      <div v-show="mode === 'adjust'" class="threshold-editor">
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
            ><strong>最小异常面积</strong
            ><small>最小连通缺陷面积 · 原图像素（px²）</small></span
          ><input
            type="number"
            min="0"
            step="1"
            v-model="draft.area_threshold"
            aria-label="最小异常面积（px²）"
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
              ...(mode === 'compare' ? [{ id: 'different', name: '判定不一致' }] : []),
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
                { id: 'binary', name: '二值图' },
                { id: 'filtered', name: '面积筛选后' },
              ]"
              :key="option.id"
              :class="{ active: view === option.id }"
              @click="view = option.id"
            >
              {{ option.name }}
            </button>
          </div>
          <div class="model-image-grid" :class="{ paired: models.length > 1 }">
            <article v-for="model in imageModels" :key="model">
              <div class="detail-heading">
                <strong>{{ names[model] }}</strong
                ><span class="badge" :class="decision(selected.id, model)">{{
                  labels[decision(selected.id, model)]
                }}</span>
              </div>
              <div class="image-viewer">
                <img
                  v-if="imageSource(model)"
                  :src="imageSource(model)"
                  :alt="`${selected.name} · ${names[model]} · ${view}`"
                />
                <p v-else>暂无此视图。二值图需要有效分数阈值及分析结果。</p>
              </div>
              <p
                v-if="resultFor(selected.id, model)?.error"
                class="inline-note invalid"
              >
                {{ resultFor(selected.id, model).error }}
              </p>
              <p v-else-if="!resultFor(selected.id, model)" class="inline-note">
                等待该模型分析完成。
              </p>
              <div class="detail-footer">
                <span
                  >分数
                  <strong>{{
                    text(resultFor(selected.id, model)?.score)
                  }}</strong></span
                ><a
                  v-if="imageSource(model)"
                  :href="imageSource(model)"
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
                {{
                  reports[model].rows.find((r) => r.id === selected.id).reason
                }}
              </p>
            </article>
          </div>
          <p class="color-note">
            二值图白色表示像素分数 ≥ 当前阈值；面积筛选后仅保留达到最小面积的 8 连通区域。图像级判定还需满足图像分数阈值。
            热力图按单张图片拉伸显示，颜色不可跨模型比较；判定使用原始分数。标注同步用于所有模型的统计，调整阈值无需重跑模型。
          </p>
        </div>
      </div>
    </template>
  </section>
</template>
