<template>
  <div class="dashboard">
    <div class="welcome">
      <h2>欢迎回来，{{ auth.user?.username ?? auth.user?.userId }}</h2>
      <p>选择一个功能开始学习</p>
    </div>

    <!-- AI 助手入口（突出展示） -->
    <el-row :gutter="16" style="margin-bottom: 16px">
      <el-col :span="24">
        <el-card
          class="ai-assistant-card"
          shadow="hover"
          @click="router.push('/chat')"
        >
          <div class="ai-card-content">
            <div class="ai-card-left">
              <span class="ai-icon">✨</span>
              <div>
                <div class="ai-title">AI 助手</div>
                <div class="ai-desc">直接描述您的需求，AI 自动识别意图并路由到最合适的 Agent — 支持多 Agent 协同工作</div>
              </div>
            </div>
            <el-button type="primary" size="default">立即体验 →</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 四个独立功能入口 -->
    <el-row :gutter="16" class="feature-cards">
      <el-col :span="6" v-for="card in featureCards" :key="card.route">
        <el-card
          class="feature-card"
          shadow="hover"
          @click="router.push(card.route)"
        >
          <div class="card-icon">{{ card.icon }}</div>
          <div class="card-title">{{ card.title }}</div>
          <div class="card-desc">{{ card.desc }}</div>
          <el-button type="primary" plain size="small" style="margin-top: 12px">
            {{ card.action }}
          </el-button>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()

const featureCards = [
  {
    icon: '🤖',
    title: '智能问答',
    desc: '7×24小时即时响应，多轮追问，RAG知识库支持',
    action: '开始问答',
    route: '/qa',
  },
  {
    icon: '📝',
    title: '试卷批改',
    desc: 'AI预批改 + 知识薄弱点分析，教师确认发布',
    action: '提交试卷',
    route: '/exam',
  },
  {
    icon: '📄',
    title: '简历审查',
    desc: '六维度写作质量评审，带原文定位的修改建议',
    action: '上传简历',
    route: '/resume',
  },
  {
    icon: '🎤',
    title: '模拟面试',
    desc: '技术原理 + 项目深挖双轨考察，结构化报告',
    action: '开始面试',
    route: '/interview',
  },
]
</script>

<style scoped>
.dashboard {
  max-width: 1100px;
}
.welcome {
  margin-bottom: 24px;
}
.welcome h2 {
  margin: 0 0 4px;
  font-size: 22px;
}
.welcome p {
  margin: 0;
  color: #8c8c8c;
}
.ai-assistant-card {
  cursor: pointer;
  background: linear-gradient(135deg, #f0f7ff 0%, #e6f4ff 100%);
  border: 1px solid #bae0ff;
  transition: transform 0.2s;
}
.ai-assistant-card:hover { transform: translateY(-2px); }
.ai-card-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 0;
}
.ai-card-left {
  display: flex;
  align-items: center;
  gap: 16px;
}
.ai-icon { font-size: 36px; }
.ai-title {
  font-size: 17px;
  font-weight: 600;
  color: #1677ff;
  margin-bottom: 4px;
}
.ai-desc {
  font-size: 13px;
  color: #595959;
}
.feature-card {
  cursor: pointer;
  text-align: center;
  padding: 8px 0;
  transition: transform 0.2s;
}
.feature-card:hover {
  transform: translateY(-2px);
}
.card-icon {
  font-size: 40px;
  margin-bottom: 12px;
}
.card-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
}
.card-desc {
  font-size: 13px;
  color: #8c8c8c;
  line-height: 1.5;
  min-height: 48px;
}
</style>
