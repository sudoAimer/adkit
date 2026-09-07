<!-- 可复用图片上传区：支持拖拽、多选、服务器缩略图与删除。 -->
<script setup>
import { ref } from "vue";
import { UploadCloud, ImagePlus, X } from "lucide-vue-next";
const props = defineProps({
  images: Array,
  disabled: Boolean,
  progress: Number,
  kind: String,
});
const emit = defineEmits(["upload", "remove"]);
const input = ref(null);
const dragging = ref(false);

function pick(files) {
  // 交给父组件统一校验和上传；立即重置输入以允许重选同名文件。
  if (!props.disabled && files?.length) emit("upload", Array.from(files));
  if (input.value) input.value.value = "";
  dragging.value = false;
}
</script>

<template>
  <div
    class="upload-zone"
    :class="{ dragging, disabled }"
    @dragover.prevent="dragging = !disabled"
    @dragleave.prevent="dragging = false"
    @drop.prevent="pick($event.dataTransfer.files)"
  >
    <input
      ref="input"
      class="sr-only"
      type="file"
      multiple
      accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
      :disabled="disabled"
      @change="pick($event.target.files)"
      :aria-label="kind === 'normal' ? '选择正常图片' : '选择待测图片'"
    />
    <div class="upload-icon"><UploadCloud :size="26" /></div>
    <strong>{{
      progress !== null ? `正在上传 ${progress}%` : "拖拽图片到这里，或点击选择"
    }}</strong>
    <p>支持 JPG、PNG、WebP、BMP、TIFF · 单张 ≤ 20 MB · 每次最多 20 张</p>
    <button
      class="button secondary"
      :disabled="disabled"
      @click="input.click()"
    >
      <ImagePlus :size="17" />选择图片
    </button>
    <progress
      v-if="progress !== null"
      :value="progress"
      max="100"
      aria-label="图片上传进度"
    ></progress>
  </div>
  <div v-if="images.length" class="sample-grid">
    <article v-for="image in images" :key="image.id" class="sample">
      <img :src="image.url" :alt="image.name" loading="lazy" />
      <button
        class="remove-image"
        :disabled="disabled"
        @click="emit('remove', image.id)"
        :aria-label="`删除 ${image.name}`"
      >
        <X :size="14" />
      </button>
      <span :title="image.name">{{ image.name }}</span>
    </article>
  </div>
</template>
