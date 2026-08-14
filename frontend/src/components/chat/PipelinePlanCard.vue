<template>
  <div class="pipeline-card">
    <!-- 标题栏 -->
    <div class="pipeline-header">
      <el-icon class="pipeline-icon"><Share /></el-icon>
      <span class="pipeline-title">{{ title }}</span>
      <el-tag type="warning" size="small" class="pipeline-tag">多 Agent 协同</el-tag>
    </div>

    <!-- 简介 -->
    <p class="pipeline-intro">{{ intro }}</p>

    <!-- 步骤列表 -->
    <div class="pipeline-steps">
      <div
        v-for="(step, index) in steps"
        :key="step.step"
        class="step-wrapper"
      >
        <!-- 步骤卡片 -->
        <div class="step-card">
          <div class="step-left">
            <div class="step-badge" :class="`step-badge--${step.agent_type}`">
              {{ agentIcon[step.agent_type] ?? '🤖' }}
            </div>
            <div class="step-info">
              <div class="step-label">
                <span class="step-num">Step {{ step.step }}</span>
                <span class="step-name">{{ step.label }}</span>
              </div>
              <div class="step-desc">{{ step.desc }}</div>
              <div class="step-tip">
                <el-icon><InfoFilled /></el-icon>
                {{ step.tip }}
              </div>
            </div>
          </div>
          <el-button
            type="primary"
            :plain="step.step > 1"
            size="small"
            class="step-btn"
            @click="router.push(step.action_url)"
          >
            {{ step.action_label }} →
          </el-button>
        </div>

        <!-- 步骤间箭头（最后一步不显示） -->
        <div v-if="index < steps.length - 1" class="step-arrow">
          <el-icon><ArrowDown /></el-icon>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import { Share, InfoFilled, ArrowDown } from '@element-plus/icons-vue'

defineProps<{
  title: string
  intro: string
  steps: Array<{
    step: number
    agent_type: string
    label: string
    desc: string
    action_label: string
    action_url: string
    tip: string
  }>
}>()

const router = useRouter()

const agentIcon: Record<string, string> = {
  resume:    '📄',
  interview: '🎤',
  exam:      '📝',
  qa:        '🤖',
}
</script>

<style scoped>
.pipeline-card {
  background: linear-gradient(135deg, #fffbe6 0%, #fff7e6 100%);
  border: 1px solid #ffd666;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 4px;
  font-size: 13px;
}

.pipeline-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  font-weight: 600;
  color: #d46b08;
}

.pipeline-icon {
  font-size: 15px;
}

.pipeline-title {
  flex: 1;
  font-size: 14px;
}

.pipeline-tag {
  font-size: 11px;
}

.pipeline-intro {
  margin: 0 0 12px;
  color: #595959;
  line-height: 1.6;
  font-size: 12.5px;
}

.pipeline-steps {
  display: flex;
  flex-direction: column;
}

.step-wrapper {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.step-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  border: 1px solid #ffe58f;
  border-radius: 6px;
  padding: 10px 12px;
  gap: 12px;
}

.step-left {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  flex: 1;
}

.step-badge {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
  background: #fff7e6;
}

.step-badge--resume    { background: #e6f4ff; }
.step-badge--interview { background: #fff0f6; }
.step-badge--exam      { background: #f6ffed; }
.step-badge--qa        { background: #f0f7ff; }

.step-info {
  display: flex;
  flex-direction: column;
  gap: 3px;
  flex: 1;
}

.step-label {
  display: flex;
  align-items: center;
  gap: 6px;
}

.step-num {
  font-size: 11px;
  color: #8c8c8c;
  background: #f5f5f5;
  padding: 1px 5px;
  border-radius: 3px;
}

.step-name {
  font-weight: 600;
  color: #262626;
  font-size: 13px;
}

.step-desc {
  color: #595959;
  font-size: 12px;
  line-height: 1.5;
}

.step-tip {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #d46b08;
  font-size: 11.5px;
}

.step-btn {
  flex-shrink: 0;
}

.step-arrow {
  display: flex;
  justify-content: center;
  padding: 4px 0;
  color: #ffa940;
  font-size: 16px;
}
</style>
