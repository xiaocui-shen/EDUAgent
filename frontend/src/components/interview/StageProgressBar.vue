<template>
  <div class="stage-bar">
    <div
      v-for="(stage, i) in stages"
      :key="stage.key"
      class="stage-item"
      :class="{
        done: stageOrder.indexOf(stage.key) < currentIndex,
        active: stage.key === current,
        pending: stageOrder.indexOf(stage.key) > currentIndex,
      }"
    >
      <div class="stage-dot">
        <el-icon v-if="stageOrder.indexOf(stage.key) < currentIndex"><Check /></el-icon>
        <span v-else>{{ i + 1 }}</span>
      </div>
      <div class="stage-label">{{ stage.label }}</div>
      <div v-if="i < stages.length - 1" class="stage-line" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Check } from '@element-plus/icons-vue'

const stages = [
  { key: 'warmup', label: '热身' },
  { key: 'tech_base', label: '技术基础' },
  { key: 'project', label: '项目深挖' },
  { key: 'closing', label: '总结' },
  { key: 'finished', label: '完成' },
]

const stageOrder = stages.map(s => s.key)

const props = defineProps<{ current: string }>()

const currentIndex = computed(() => stageOrder.indexOf(props.current))
</script>

<style scoped>
.stage-bar {
  display: flex;
  align-items: center;
  padding: 12px 0;
  position: relative;
}
.stage-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  flex: 1;
}
.stage-dot {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 500;
  border: 2px solid #d9d9d9;
  background: #fff;
  color: #8c8c8c;
  z-index: 1;
}
.stage-item.done .stage-dot {
  background: #52c41a;
  border-color: #52c41a;
  color: #fff;
}
.stage-item.active .stage-dot {
  background: #1677ff;
  border-color: #1677ff;
  color: #fff;
}
.stage-label {
  font-size: 12px;
  margin-top: 4px;
  color: #8c8c8c;
}
.stage-item.active .stage-label { color: #1677ff; font-weight: 500; }
.stage-item.done .stage-label { color: #52c41a; }
.stage-line {
  position: absolute;
  top: 14px;
  left: 50%;
  width: 100%;
  height: 2px;
  background: #d9d9d9;
  z-index: 0;
}
.stage-item.done .stage-line { background: #52c41a; }
</style>
