<template>
  <div class="chat-page">
    <div class="chat-panel">
      <!-- 顶部说明 -->
      <div class="chat-header-hint">
        <el-icon><MagicStick /></el-icon>
        AI 助手会自动识别您的需求，路由到最合适的 Agent 为您服务
      </div>

      <!-- 消息区 -->
      <div class="chat-messages" ref="messagesEl">
        <div v-if="messages.length === 0 && !isStreaming" class="empty-hint">
          <p>✨ 您好！我是 EduAgent AI 助手</p>
          <p>请直接告诉我您的需求，例如：</p>
          <div class="example-queries">
            <el-tag
              v-for="q in exampleQueries"
              :key="q"
              class="example-tag"
              @click="sendExample(q)"
            >{{ q }}</el-tag>
          </div>
        </div>

        <template v-for="(msg, i) in messages" :key="i">
          <!-- 路由决策卡片（仅 assistant 消息有） -->
          <RoutingDecisionCard
            v-if="msg.routingDecision"
            :agent-type="msg.routingDecision.agent_type"
            :agent-display="msg.routingDecision.agent_display"
            :confidence="msg.routingDecision.confidence"
            :reason="msg.routingDecision.reason"
            :execution-mode="msg.routingDecision.execution_mode"
          />

          <!-- 多 Agent 协同管道计划卡片 -->
          <PipelinePlanCard
            v-if="msg.pipelinePlan"
            :title="msg.pipelinePlan.title"
            :intro="msg.pipelinePlan.intro"
            :steps="msg.pipelinePlan.steps"
          />

          <!-- 普通气泡（pipeline_plan 消息无需气泡） -->
          <ChatBubble v-if="!msg.pipelinePlan" :role="msg.role" :sources="msg.sources">
            <!-- 引导消息：带跳转按钮 -->
            <template v-if="msg.guidance">
              <p style="margin: 0 0 10px">{{ msg.guidance.message }}</p>
              <el-button
                v-if="msg.guidance.action_label"
                type="primary"
                size="small"
                @click="router.push(msg.guidance.action_url)"
              >
                {{ msg.guidance.action_label }} →
              </el-button>
            </template>
            <MarkdownRenderer v-else :content="msg.content" />
          </ChatBubble>
        </template>

        <!-- 流式气泡 -->
        <template v-if="isStreaming">
          <RoutingDecisionCard
            v-if="streamingRouting"
            :agent-type="streamingRouting.agent_type"
            :agent-display="streamingRouting.agent_display"
            :confidence="streamingRouting.confidence"
            :reason="streamingRouting.reason"
            :execution-mode="streamingRouting.execution_mode"
          />
          <ChatBubble role="assistant">
            <template v-if="streamingText">
              <MarkdownRenderer :content="streamingText" />
            </template>
            <span v-else-if="progressStage" class="progress-hint">
              <span class="dot" /><span class="dot" /><span class="dot" />
              {{ progressStage }}
            </span>
            <span v-else class="thinking">
              <span class="dot" /><span class="dot" /><span class="dot" />
            </span>
          </ChatBubble>
        </template>
      </div>

      <!-- 输入区 -->
      <div class="chat-input-area">
        <el-input
          ref="inputRef"
          v-model="inputText"
          type="textarea"
          :rows="3"
          placeholder="直接描述您的需求，Enter 发送，Shift+Enter 换行"
          resize="none"
          :disabled="isStreaming"
          @keydown="handleKeydown"
        />
        <div class="input-actions">
          <el-tag v-if="lastAnswerMode" size="small" :type="answerModeTagType">
            {{ answerModeLabel }}
          </el-tag>
          <el-button
            type="primary"
            :loading="isStreaming"
            :disabled="!inputText.trim() || isStreaming"
            @click="sendMessage"
          >
            发送
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { MagicStick } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import ChatBubble from '@/components/chat/ChatBubble.vue'
import MarkdownRenderer from '@/components/chat/MarkdownRenderer.vue'
import RoutingDecisionCard from '@/components/chat/RoutingDecisionCard.vue'
import PipelinePlanCard from '@/components/chat/PipelinePlanCard.vue'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()

interface RoutingDecision {
  agent_type: string
  agent_display: string
  confidence: number
  reason: string
  execution_mode: string
}

interface GuidanceInfo {
  message: string
  action_label: string
  action_url: string
}

interface PipelineStep {
  step: number
  agent_type: string
  label: string
  desc: string
  action_label: string
  action_url: string
  tip: string
}

interface PipelinePlan {
  title: string
  intro: string
  steps: PipelineStep[]
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  routingDecision?: RoutingDecision
  guidance?: GuidanceInfo
  pipelinePlan?: PipelinePlan
}

const messages = ref<Message[]>([])
const inputText = ref('')
const messagesEl = ref<HTMLElement>()
// 固定 session：同一用户始终复用同一 thread，避免 MemorySaver 无限累积
const sessionId = ref(`unified_${auth.user?.userId ?? 'guest'}`)

const isStreaming = ref(false)
const streamingText = ref('')
const streamingRouting = ref<RoutingDecision | null>(null)
const progressStage = ref('')
const lastAnswerMode = ref('')
const lastConfidence = ref(0)
const lastSources = ref<string[]>([])

const exampleQueries = [
  'Python 中的 GIL 是什么？',
  '我想提交试卷批改',
  '帮我审查一下简历',
  '我要开始模拟面试',
  '帮我准备求职',
]

const answerModeTagType = computed(() => {
  if (lastAnswerMode.value === 'rag') return 'success'
  if (lastAnswerMode.value === 'general') return 'warning'
  return 'info'
})

const answerModeLabel = computed(() => {
  const map: Record<string, string> = {
    rag: `RAG ${lastConfidence.value}%`,
    general: '通用回答',
    llm_direct: 'LLM直答',
  }
  return map[lastAnswerMode.value] ?? lastAnswerMode.value
})

function sendExample(q: string) {
  inputText.value = q
  sendMessage()
}

function handleKeydown(e: KeyboardEvent) {
  if (e.isComposing) return
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || isStreaming.value) return

  inputText.value = ''
  messages.value.push({ role: 'user', content: text })
  await scrollToBottom()

  isStreaming.value = true
  streamingText.value = ''
  streamingRouting.value = null
  progressStage.value = ''
  lastAnswerMode.value = ''
  lastSources.value = []

  let pendingRouting: RoutingDecision | null = null
  let pendingGuidance: GuidanceInfo | null = null
  let pendingPipelinePlan: PipelinePlan | null = null

  try {
    const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'
    const resp = await fetch(`${apiBase}/api/v1/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth.token}`,
      },
      body: JSON.stringify({
        session_id: sessionId.value,
        message: text,
      }),
    })

    if (resp.status === 401) {
      localStorage.removeItem('edu-agent-token')
      localStorage.removeItem('edu-agent-user')
      router.push('/login')
      return
    }
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data:')) continue
        const raw = line.slice(5).trim()
        if (!raw) continue

        try {
          const evt = JSON.parse(raw)

          if (evt.type === 'routing_decision') {
            pendingRouting = {
              agent_type:     evt.agent_type,
              agent_display:  evt.agent_display,
              confidence:     evt.confidence,
              reason:         evt.reason,
              execution_mode: evt.execution_mode,
            }
            // 立刻显示路由卡片（流式气泡上方）
            streamingRouting.value = pendingRouting
            await scrollToBottom()

          } else if (evt.type === 'progress') {
            progressStage.value = evt.stage

          } else if (evt.type === 'token') {
            progressStage.value = ''
            streamingText.value += evt.content
            await scrollToBottom()

          } else if (evt.type === 'guidance') {
            pendingGuidance = {
              message:      evt.message,
              action_label: evt.action_label ?? '',
              action_url:   evt.action_url ?? '',
            }

          } else if (evt.type === 'pipeline_plan') {
            pendingPipelinePlan = {
              title: evt.title,
              intro: evt.intro,
              steps: evt.steps ?? [],
            }
            await scrollToBottom()

          } else if (evt.type === 'meta') {
            lastAnswerMode.value = evt.answer_mode ?? ''
            lastConfidence.value = Math.round((evt.confidence ?? 0) * 100)
            lastSources.value    = evt.sources ?? []

          } else if (evt.type === 'error') {
            throw new Error(evt.message ?? 'SSE error')
          }
        } catch {
          // 忽略非 JSON 行
        }
      }
    }

    // 流结束：将流式内容固化为消息
    if (pendingPipelinePlan) {
      messages.value.push({
        role:            'assistant',
        content:         '',
        routingDecision: pendingRouting ?? undefined,
        pipelinePlan:    pendingPipelinePlan,
      })
    } else if (pendingGuidance) {
      messages.value.push({
        role:            'assistant',
        content:         pendingGuidance.message,
        routingDecision: pendingRouting ?? undefined,
        guidance:        pendingGuidance,
      })
    } else if (streamingText.value) {
      messages.value.push({
        role:            'assistant',
        content:         streamingText.value,
        sources:         lastSources.value,
        routingDecision: pendingRouting ?? undefined,
      })
    }

  } catch (err) {
    ElMessage.error('请求失败，请重试')
    console.error('[UnifiedChat SSE]', err)
  } finally {
    isStreaming.value = false
    streamingText.value = ''
    streamingRouting.value = null
    progressStage.value = ''
    await scrollToBottom()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}
</script>

<style scoped>
.chat-page {
  display: flex;
  height: calc(100vh - 56px - 48px);
}
.chat-panel {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chat-header-hint {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  font-size: 13px;
  color: #8c8c8c;
  border-bottom: 1px solid #f0f0f0;
  background: #fafafa;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.empty-hint {
  text-align: center;
  color: #8c8c8c;
  margin-top: 60px;
  font-size: 15px;
  line-height: 2;
}
.example-queries {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin-top: 12px;
}
.example-tag {
  cursor: pointer;
  transition: opacity 0.15s;
}
.example-tag:hover { opacity: 0.75; }
.chat-input-area {
  border-top: 1px solid #f0f0f0;
  padding: 12px 16px;
}
.input-actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
.thinking, .progress-hint {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 0;
  font-size: 13px;
  color: #8c8c8c;
}
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #bfbfbf;
  animation: bounce 1.2s infinite ease-in-out;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
  40%            { transform: scale(1.1); opacity: 1; }
}
</style>
