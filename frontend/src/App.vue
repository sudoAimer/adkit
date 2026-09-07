<!-- 单人检测工作台，按正常样本、建库、结果三个阶段组织操作。 -->
<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  Activity,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  FolderOpen,
  Image,
  Layers3,
  LoaderCircle,
  Plus,
  RefreshCw,
  ScanLine,
  SlidersHorizontal,
  Trash2,
  X,
  Download,
  AlertCircle,
} from "lucide-vue-next";
import ImageUpload from "./components/ImageUpload.vue";
import { request, uploadImages } from "./api";

const tasks = ref([]);
const algorithms = ref([]);
const selectedId = ref("");
const stage = ref("normal");
const error = ref("");
const notice = ref("");
const loading = ref(true);
const busy = ref(false);
const uploadProgress = ref(null);
const modal = ref(null);
const newName = ref("");
const newAlgorithm = ref("anomalydino");
const newSize = ref(448);
const newRotation = ref(false);
const thresholdDraft = ref("");
const resultId = ref("");
const view = ref("overlay");
const filter = ref("all");
let timer;
let stopped = false;
let refreshSequence = 0;
const current = computed(() =>
  tasks.value.find((task) => task.id === selectedId.value),
);
const running = computed(() =>
  ["queued", "running"].includes(current.value?.job.state),
);
const disabled = computed(() => busy.value || running.value);
const results = computed(() => current.value?.results || []);
const effectiveThreshold = computed(() =>
  thresholdDraft.value === "" ? null : Number(thresholdDraft.value),
);
const validThreshold = computed(
  () =>
    effectiveThreshold.value === null ||
    (Number.isFinite(effectiveThreshold.value) &&
      effectiveThreshold.value >= 0),
);
const thresholdChanged = computed(
  () => effectiveThreshold.value !== current.value?.threshold,
);
const scored = computed(() =>
  results.value.filter((item) => Number.isFinite(item.score)),
);
const defectCount = computed(
  () => scored.value.filter((item) => classification(item) === "defect").length,
);
const filteredResults = computed(() =>
  results.value.filter(
    (item) => filter.value === "all" || classification(item) === filter.value,
  ),
);
const selectedResult = computed(
  () =>
    filteredResults.value.find((item) => item.id === resultId.value) ||
    filteredResults.value[0],
);
const sliderMax = computed(() =>
  Math.max(
    0.1,
    ...scored.value.map((item) => item.score * 1.2),
    current.value?.threshold || 0,
    Number(thresholdDraft.value) || 0,
  ),
);
const statusText = {
  idle: "待处理",
  queued: "排队中",
  running: "处理中",
  done: "已完成",
  failed: "处理失败",
};

function classification(item) {
  // 原始图像分数达到阈值即判为缺陷；未设置或输入无效时不作结论。
  if (item.error) return "error";
  if (effectiveThreshold.value === null || !validThreshold.value)
    return "unset";
  return item.score >= effectiveThreshold.value ? "defect" : "normal";
}

function label(item) {
  // 将内部判定状态映射成中文标签。
  return { defect: "缺陷", normal: "正常", unset: "未判定", error: "检测失败" }[
    classification(item)
  ];
}

function scoreText(score) {
  // 同时保留很小和很大的原始分数，避免显示为错误的零。
  return Number.isFinite(score) ? Number(score).toPrecision(5) : "—";
}

function selectTask(task) {
  // 切换任务时恢复该任务保存的阈值，不沿用另一任务的分数尺度。
  selectedId.value = task.id;
  thresholdDraft.value = task.threshold ?? "";
  resultId.value = "";
  filter.value = "all";
  stage.value = task.results.length
    ? "results"
    : task.model_ready
      ? "fit"
      : "normal";
  error.value = "";
  notice.value = "";
}

async function refresh() {
  // 序号保证较早发出的轮询不会覆盖较新的操作结果。
  const sequence = ++refreshSequence;
  const data = await request("/tasks");
  if (sequence !== refreshSequence) return;
  tasks.value = data;
  if (!data.some((task) => task.id === selectedId.value)) {
    if (data.length) selectTask(data[0]);
    else selectedId.value = "";
  }
}

async function perform(action) {
  // 所有交互统一管理忙碌与错误状态，避免重复提交。
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  notice.value = "";
  try {
    await action();
    await refresh();
  } catch (exc) {
    error.value = exc.message;
  } finally {
    busy.value = false;
  }
}

function openCreate() {
  // 使用原生对话框获得焦点约束和 Escape 关闭支持。
  newName.value = "";
  modal.value.showModal();
}

function createTask() {
  // 提交明确的算法参数，成功后进入空任务的正常图片步骤。
  perform(async () => {
    const task = await request("/tasks", {
      method: "POST",
      body: JSON.stringify({
        name: newName.value,
        algorithm: newAlgorithm.value,
        image_size: Number(newSize.value),
        rotation: newRotation.value,
      }),
    });
    tasks.value.unshift(task);
    selectTask(task);
    modal.value.close();
  });
}

function changeAlgorithm() {
  // 新任务切换算法时使用该算法推荐的默认输入尺寸。
  newSize.value = newAlgorithm.value === "subspacead" ? 672 : 448;
}

function upload(kind, files) {
  // 上传前检查常见限制，服务端仍执行完整格式与尺寸校验。
  if (files.length > 20 || files.some((file) => file.size > 20 * 1024 * 1024)) {
    error.value = "每次最多上传 20 张图片，单张不能超过 20 MB。";
    return;
  }
  const id = selectedId.value;
  perform(async () => {
    uploadProgress.value = 0;
    try {
      await uploadImages(id, kind, files, (value) => {
        uploadProgress.value = value;
      });
    } finally {
      uploadProgress.value = null;
    }
  });
}

function removeImage(kind, id) {
  // 删除样本后，服务端统一使旧模型或结果失效。
  const taskId = selectedId.value;
  perform(() =>
    request(`/tasks/${taskId}/images/${kind}/${id}`, { method: "DELETE" }),
  );
}

function start(operation) {
  // 建库只显示阶段状态；检测按实际完成图片数展示进度。
  const id = selectedId.value;
  perform(async () => {
    await request(`/tasks/${id}/jobs/${operation}`, { method: "POST" });
    if (id === selectedId.value)
      stage.value = operation === "fit" ? "fit" : "results";
  });
}

function saveThreshold() {
  // 只持久化判定阈值，不修改热力图或重跑检测。
  if (!validThreshold.value) return;
  const id = selectedId.value;
  const threshold = effectiveThreshold.value;
  perform(async () => {
    await request(`/tasks/${id}/threshold`, {
      method: "PATCH",
      body: JSON.stringify({ threshold }),
    });
    notice.value = "阈值已保存";
  });
}

function deleteTask() {
  // 明确确认后删除整项任务及其图片、参考库和结果。
  const task = current.value;
  if (
    !window.confirm(
      `删除「${task.name}」？其图片、参考库和检测结果也会被删除。`,
    )
  )
    return;
  perform(() => request(`/tasks/${task.id}`, { method: "DELETE" }));
}

async function poll() {
  // 串行轮询避免请求堆积，连接恢复后自动更新后台作业状态。
  try {
    await refresh();
  } catch (exc) {
    error.value = exc.message;
  }
  if (!stopped) timer = window.setTimeout(poll, 2000);
}

onMounted(async () => {
  // 首次加载算法资源信息和任务列表，然后开始状态轮询。
  try {
    const settings = await request("/settings");
    algorithms.value = settings.algorithms;
    await refresh();
  } catch (exc) {
    error.value = exc.message;
  } finally {
    loading.value = false;
    if (!stopped) timer = window.setTimeout(poll, 2000);
  }
});
onUnmounted(() => {
  stopped = true;
  window.clearTimeout(timer);
});
</script>

<template>
  <div class="workbench">
    <aside class="sidebar">
      <a href="/" class="brand"
        ><span class="brand-icon"><ScanLine :size="25" /></span
        ><span>adkit<span class="brand-caption">图像检测工作台</span></span></a
      >
      <div class="workspace-label">个人工作区 <span>单人模式</span></div>
      <button
        class="button primary new-task"
        @click="openCreate"
        :disabled="busy"
      >
        <Plus :size="18" />新建检测任务
      </button>
      <div class="list-label">
        <span>检测任务</span><span>{{ tasks.length }}</span>
      </div>
      <nav class="task-nav" aria-label="检测任务">
        <button
          v-for="task in tasks"
          :key="task.id"
          class="task-button"
          :class="{ active: task.id === selectedId }"
          :disabled="busy"
          @click="selectTask(task)"
        >
          <FolderOpen :size="19" /><span class="task-info"
            ><strong>{{ task.name }}</strong
            ><small
              >{{
                task.algorithm === "anomalydino" ? "AnomalyDINO" : "SubspaceAD"
              }}
              · {{ task.normal.length }} 张正常样本</small
            ></span
          ><LoaderCircle
            v-if="['queued', 'running'].includes(task.job.state)"
            :size="15"
            class="spin"
          />
        </button>
        <p v-if="!tasks.length" class="nav-empty">还没有检测任务</p>
      </nav>
      <div class="sidebar-foot">
        <Layers3 :size="18" />
        <div>正常样本建库<small>无需标注缺陷区域</small></div>
      </div>
    </aside>

    <main>
      <header class="topbar">
        <span
          >工作区 <ChevronRight :size="14" />{{
            current?.name || "新建检测任务"
          }}</span
        ><span class="topbar-meta">本地模型 · 图片检测</span>
      </header>
      <div class="content">
        <div v-if="error" class="feedback error" role="alert">
          <AlertCircle :size="19" /><span>{{ error }}</span
          ><button @click="error = ''" aria-label="关闭提示">
            <X :size="16" />
          </button>
        </div>
        <div v-if="notice" class="feedback success" role="status">
          <CheckCircle2 :size="18" />{{ notice }}
        </div>
        <div v-if="loading" class="empty-state">
          <LoaderCircle class="spin" />正在加载工作区…
        </div>
        <template v-else-if="current">
          <div class="page-heading">
            <div>
              <p class="eyebrow">图像异常检测</p>
              <h1>{{ current.name }}</h1>
              <p>
                {{
                  current.algorithm === "anomalydino"
                    ? "AnomalyDINO"
                    : "SubspaceAD"
                }}<span class="separator">/</span
                >{{ current.image_size }} px<span class="separator">/</span
                >{{ current.normal.length }} 张正常样本
              </p>
            </div>
            <button
              class="icon-button"
              @click="deleteTask"
              :disabled="disabled"
              aria-label="删除当前任务"
              title="删除任务"
            >
              <Trash2 :size="19" />
            </button>
          </div>
          <div class="steps" aria-label="操作步骤">
            <button
              v-for="(step, index) in [
                { id: 'normal', title: '正常样本', sub: '上传无缺陷图片' },
                { id: 'fit', title: '建立参考库', sub: '从正常样本学习' },
                { id: 'results', title: '检测与结果', sub: '查看异常分布' },
              ]"
              :key="step.id"
              :class="{ active: stage === step.id }"
              :disabled="busy"
              @click="stage = step.id"
            >
              <span class="step-number"
                ><Check
                  v-if="
                    (index === 0 && current.normal.length) ||
                    (index === 1 && current.model_ready)
                  "
                  :size="18"
                /><template v-else>{{ index + 1 }}</template></span
              ><span
                ><strong>{{ step.title }}</strong
                ><small>{{ step.sub }}</small></span
              ><ChevronRight :size="18" class="step-chevron" />
            </button>
          </div>
          <div
            v-if="running || current.job.state === 'failed'"
            class="job-banner"
            :class="{ failed: current.job.state === 'failed' }"
            role="status"
            aria-live="polite"
          >
            <LoaderCircle v-if="running" class="spin" :size="20" /><AlertCircle
              v-else
              :size="20"
            />
            <div>
              <strong>{{ statusText[current.job.state] }}</strong>
              <p>{{ current.job.message }}</p>
              <progress
                v-if="current.job.total"
                :value="current.job.completed"
                :max="current.job.total"
              ></progress>
            </div>
          </div>

          <section v-if="stage === 'normal'" class="panel">
            <div class="section-heading">
              <div>
                <h2>
                  上传正常样本
                  <span class="count">{{ current.normal.length }}</span>
                </h2>
                <p>选择同一类物品、无缺陷的图片，用于建立正常外观参考库。</p>
              </div>
              <span class="quiet-tag">正常图片</span>
            </div>
            <ImageUpload
              :images="current.normal"
              :disabled="disabled"
              :progress="uploadProgress"
              kind="normal"
              @upload="upload('normal', $event)"
              @remove="removeImage('normal', $event)"
            />
            <div class="panel-footer">
              <p><CircleHelp :size="16" />增删正常图片后，需要重新建库。</p>
              <button
                class="button primary"
                :disabled="!current.normal.length || disabled"
                @click="stage = 'fit'"
              >
                下一步：建立参考库<ArrowRight :size="17" />
              </button>
            </div>
          </section>

          <section v-if="stage === 'fit'" class="panel fit-panel">
            <div class="section-heading">
              <div>
                <h2>建立正常参考库</h2>
                <p>模型提取正常样本的特征，作为后续缺陷检测的依据。</p>
              </div>
              <span class="quiet-tag" :class="{ ready: current.model_ready }">{{
                current.model_ready ? "参考库已就绪" : "等待建库"
              }}</span>
            </div>
            <div class="fit-summary">
              <div>
                <Image :size="22" /><strong
                  >{{ current.normal.length }}<small>张</small></strong
                ><span>正常样本</span>
              </div>
              <div>
                <Layers3 :size="22" /><strong>{{
                  current.algorithm === "anomalydino"
                    ? "AnomalyDINO"
                    : "SubspaceAD"
                }}</strong
                ><span>检测算法</span>
              </div>
              <div>
                <ScanLine :size="22" /><strong
                  >{{ current.image_size }}<small>px</small></strong
                ><span>输入尺寸</span>
              </div>
            </div>
            <div class="fit-status">
              <span class="fit-status-icon"
                ><CheckCircle2
                  v-if="current.model_ready"
                  :size="36" /><LoaderCircle
                  v-else-if="running"
                  :size="36"
                  class="spin" /><Layers3 v-else :size="36"
              /></span>
              <h3>
                {{
                  current.model_ready
                    ? "参考库已准备好"
                    : running
                      ? "正在建立参考库"
                      : "准备开始建库"
                }}
              </h3>
              <p>
                {{
                  current.model_ready
                    ? "上传待测图片，查看模型发现的异常区域。"
                    : "建库耗时取决于样本数量与设备性能，请耐心等待。"
                }}
              </p>
              <p
                v-if="
                  algorithms.find((item) => item.id === current.algorithm)
                    ?.ready === false
                "
                class="resource-note"
              >
                服务器尚未准备该算法的本地权重，请先完成配置。
              </p>
              <div class="button-row">
                <button
                  class="button"
                  :class="current.model_ready ? 'secondary' : 'primary'"
                  @click="start('fit')"
                  :disabled="disabled || !current.normal.length"
                >
                  <RefreshCw v-if="current.model_ready" :size="17" /><Layers3
                    v-else
                    :size="17"
                  />{{ current.model_ready ? "重新建库" : "开始建库" }}</button
                ><button
                  v-if="current.model_ready"
                  class="button primary"
                  @click="stage = 'results'"
                >
                  开始检测<ArrowRight :size="17" />
                </button>
              </div>
            </div>
          </section>

          <template v-if="stage === 'results'">
            <section class="panel">
              <div class="section-heading">
                <div>
                  <h2>
                    待测图片
                    <span class="count">{{ current.test.length }}</span>
                  </h2>
                  <p>上传需要检查的图片，支持批量检测。</p>
                </div>
                <button
                  class="button primary"
                  @click="start('predict')"
                  :disabled="
                    disabled || !current.model_ready || !current.test.length
                  "
                >
                  <ScanLine :size="17" />{{
                    results.length ? "重新检测" : "开始检测"
                  }}
                </button>
              </div>
              <p v-if="!current.model_ready" class="inline-note">
                请先完成正常样本建库，再开始检测。
              </p>
              <ImageUpload
                :images="current.test"
                :disabled="disabled"
                :progress="uploadProgress"
                kind="test"
                @upload="upload('test', $event)"
                @remove="removeImage('test', $event)"
              />
            </section>
            <section class="panel results-panel">
              <div class="section-heading">
                <div>
                  <h2>检测结果</h2>
                  <p>原始异常分数越高，表示与正常样本差异越大。</p>
                </div>
                <span class="quiet-tag">{{ scored.length }} 张已完成</span>
              </div>
              <div class="threshold-bar">
                <div class="threshold-title">
                  <SlidersHorizontal :size="20" />
                  <div>
                    <strong>图像判定阈值</strong
                    ><small>分数 ≥ 阈值时判为缺陷</small>
                  </div>
                </div>
                <div class="threshold-controls">
                  <input
                    aria-label="拖动调整图像阈值"
                    type="range"
                    min="0"
                    :max="sliderMax"
                    :step="sliderMax / 1000"
                    :value="effectiveThreshold ?? 0"
                    @input="thresholdDraft = $event.target.value"
                  /><input
                    aria-label="图像判定阈值"
                    class="threshold-number"
                    type="number"
                    min="0"
                    step="any"
                    placeholder="未设置"
                    v-model="thresholdDraft"
                  /><button
                    class="button secondary"
                    :disabled="busy || !validThreshold || !thresholdChanged"
                    @click="saveThreshold"
                  >
                    保存阈值</button
                  ><button
                    class="text-button"
                    :disabled="busy"
                    @click="thresholdDraft = ''"
                  >
                    清空
                  </button>
                </div>
              </div>
              <p class="threshold-note" :class="{ invalid: !validThreshold }">
                {{
                  !validThreshold
                    ? "请输入大于或等于 0 的有限数值。"
                    : thresholdChanged
                      ? "当前是阈值预览，点击“保存阈值”后保留。"
                      : "请结合正常与缺陷样本设置阈值。异常分数不是概率，不同任务的阈值不通用。"
                }}
              </p>
              <div v-if="!results.length" class="result-empty">
                <ScanLine :size="32" />
                <h3>检测结果将在这里显示</h3>
                <p>完成建库并提交待测图片后，可查看分数、热力图与叠加图。</p>
              </div>
              <template v-else>
                <div class="results-toolbar">
                  <div class="filter-tabs" aria-label="结果筛选">
                    <button
                      :class="{ active: filter === 'all' }"
                      @click="filter = 'all'"
                    >
                      全部 {{ results.length }}</button
                    ><button
                      :class="{ active: filter === 'defect' }"
                      @click="filter = 'defect'"
                    >
                      缺陷 {{ defectCount }}
                    </button>
                  </div>
                  <span>判定随阈值实时更新</span>
                </div>
                <div class="result-workspace">
                  <div class="result-list">
                    <button
                      v-for="item in filteredResults"
                      :key="item.id"
                      :class="{ active: selectedResult?.id === item.id }"
                      @click="resultId = item.id"
                    >
                      <img
                        :src="item.original"
                        :alt="item.name"
                        loading="lazy"
                      /><span
                        ><strong>{{ item.name }}</strong
                        ><small>分数 {{ scoreText(item.score) }}</small></span
                      ><em class="badge" :class="classification(item)">{{
                        label(item)
                      }}</em>
                    </button>
                    <p v-if="!filteredResults.length" class="nav-empty">
                      当前阈值下没有缺陷图片
                    </p>
                  </div>
                  <div v-if="selectedResult" class="result-detail">
                    <div class="detail-heading">
                      <strong :title="selectedResult.name">{{
                        selectedResult.name
                      }}</strong
                      ><span
                        class="badge"
                        :class="classification(selectedResult)"
                        >{{ label(selectedResult) }}</span
                      >
                    </div>
                    <div class="view-tabs" aria-label="图片视图">
                      <button
                        v-for="option in [
                          { id: 'original', title: '原图' },
                          { id: 'heatmap', title: '热力图' },
                          { id: 'overlay', title: '叠加图' },
                        ]"
                        :key="option.id"
                        :class="{ active: view === option.id }"
                        @click="view = option.id"
                        :disabled="
                          !!selectedResult.error && option.id !== 'original'
                        "
                      >
                        {{ option.title }}
                      </button>
                    </div>
                    <div class="image-viewer">
                      <img
                        :src="
                          selectedResult.error
                            ? selectedResult.original
                            : selectedResult[view]
                        "
                        :alt="`${selectedResult.name} · ${view === 'original' ? '原图' : view === 'heatmap' ? '热力图' : '叠加图'}`"
                      />
                    </div>
                    <p v-if="selectedResult.error" class="inline-note invalid">
                      {{ selectedResult.error }}
                    </p>
                    <div class="detail-footer">
                      <span
                        >异常分数
                        <strong>{{
                          scoreText(selectedResult.score)
                        }}</strong></span
                      ><a
                        :href="
                          selectedResult.error
                            ? selectedResult.original
                            : selectedResult[view]
                        "
                        :download="`${selectedResult.id}-${view}.png`"
                        class="text-button"
                        ><Download :size="16" />下载图片</a
                      >
                    </div>
                    <p class="color-note">
                      热力图颜色按单张图片拉伸，仅展示异常分布；调整图像阈值不会改变热力图。
                    </p>
                  </div>
                </div>
              </template>
            </section>
          </template>
        </template>
        <section v-else class="welcome panel">
          <div class="welcome-symbol"><ScanLine :size="40" /></div>
          <p class="eyebrow">ADKIT WORKSPACE</p>
          <h1>从正常样本开始检测</h1>
          <p>建立物品的正常外观参考库，发现待测图片中的异常。</p>
          <div class="welcome-steps">
            <span><Image :size="22" />上传正常图片</span
            ><ChevronRight :size="18" /><span
              ><Layers3 :size="22" />在线建立参考库</span
            ><ChevronRight :size="18" /><span
              ><Activity :size="22" />查看缺陷结果</span
            >
          </div>
          <button class="button primary" @click="openCreate">
            <Plus :size="18" />新建第一个任务
          </button>
        </section>
        <footer class="page-footer">
          adkit<span>正常样本建库 · 图像异常检测</span>
        </footer>
      </div>
    </main>
    <dialog
      ref="modal"
      class="create-dialog"
      @click="$event.target === modal && modal.close()"
    >
      <form @submit.prevent="createTask">
        <div class="section-heading">
          <h2>新建检测任务</h2>
          <button
            type="button"
            class="icon-button"
            @click="modal.close()"
            aria-label="关闭"
          >
            <X :size="20" />
          </button>
        </div>
        <label
          >任务名称<input
            autofocus
            required
            maxlength="60"
            placeholder="例如：瓶身外观检测"
            v-model.trim="newName" /></label
        ><label
          >检测算法<select v-model="newAlgorithm" @change="changeAlgorithm">
            <option value="anomalydino">AnomalyDINO</option>
            <option value="subspacead">SubspaceAD</option>
          </select></label
        >
        <p class="form-hint">
          同一任务建议只检测一类物品，保持拍摄角度与光照接近。
        </p>
        <details>
          <summary>高级设置</summary>
          <label
            >输入尺寸<select v-model="newSize">
              <option :value="224">224 px</option>
              <option :value="448">448 px</option>
              <option :value="672">672 px</option>
              <option :value="896">896 px</option>
            </select></label
          ><label class="checkbox-label"
            ><input
              type="checkbox"
              v-model="newRotation"
            />正常图片旋转增强</label
          >
          <p class="form-hint">
            创建后算法和输入尺寸固定；需要更换时请新建任务。
          </p>
        </details>
        <p v-if="error" class="inline-note invalid" role="alert">{{ error }}</p>
        <div class="dialog-actions">
          <button type="button" class="button secondary" @click="modal.close()">
            取消</button
          ><button class="button primary" :disabled="busy || !newName">
            {{ busy ? "创建中…" : "创建任务" }}<ArrowRight :size="16" />
          </button>
        </div>
      </form>
    </dialog>
  </div>
</template>
