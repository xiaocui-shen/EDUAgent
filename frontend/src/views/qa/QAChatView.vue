<template>
  <div class="qa-page">
    <!-- 左侧：会话列表 -->
    <div class="session-panel">
      <div class="session-header">
        <span>会话列表</span>
        <el-button :icon="Plus" circle size="small" @click="newSession" />
      </div>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          class="session-item"
          :class="{ active: s.id === currentSessionId }"
          @click="switchSession(s.id)"
        >
          <el-icon><ChatDotRound /></el-icon>
          <span class="session-name">{{ s.name }}</span>
          <el-icon class="delete-btn" @click.stop="deleteSession(s.id)"><Close /></el-icon>
        </div>
      </div>
    </div>

    <!-- 右侧：聊天区 -->
    <div class="chat-panel">
      <div class="chat-messages" ref="messagesEl">
        <div v-if="messages.length === 0 && !isStreaming" class="empty-hint">
          <p>🤖 你好！我是 EduAgent 智能助手</p>
          <p>有任何 IT 学习问题，随时问我</p>
        </div>

        <ChatBubble
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :sources="msg.sources"
        >
          <MarkdownRenderer :content="msg.content" />
        </ChatBubble>

        <!-- 流式气泡：isStreaming 期间始终显示，token 到达前显示进度/跳点动画 -->
        <ChatBubble v-if="isStreaming" role="assistant">
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
      </div>

      <div class="chat-input-area">
        <el-input
          ref="inputRef"
          v-model="inputText"
          type="textarea"
          :rows="3"
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          resize="none"
          :disabled="isStreaming"
          @keydown="handleKeydown"
        />
        <div class="input-actions">
          <el-tooltip
            :content="webSearchEnabled ? 'Web Search 已开启：低置信度时先搜索互联网再回答' : 'Web Search 已关闭：低置信度时直接由大模型回答'"
            placement="top"
          >
            <el-switch
              v-model="webSearchEnabled"
              :disabled="isStreaming"
              active-text="🌐"
              inactive-text=""
              size="small"
            />
          </el-tooltip>
          <el-tag
            v-if="lastAnswerMode"
            size="small"
            :type="answerModeTagType"
          >
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
import { ref, computed, nextTick, onMounted } from 'vue'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
import { useRouter } from 'vue-router'
import { Plus, ChatDotRound, Close } from '@element-plus/icons-vue'
import { v4 as uuidv4 } from 'uuid'
import { ElMessage } from 'element-plus'
import ChatBubble from '@/components/chat/ChatBubble.vue'
import MarkdownRenderer from '@/components/chat/MarkdownRenderer.vue'
import { useAuthStore } from '@/stores/auth'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
}

interface Session {
  id: string
  name: string
  messages: Message[]
}

const router = useRouter()
const auth = useAuthStore()

const sessions = ref<Session[]>([])
const currentSessionId = ref('')
const messages = ref<Message[]>([])
const inputText = ref('')
const messagesEl = ref<HTMLElement>()

// 流式状态（本地管理，不用 composable，避免 proxy 缓冲问题）
const isStreaming = ref(false)
const streamingText = ref('')
const progressStage = ref('')   // 进度提示文字
const lastAnswerMode = ref('')
const lastConfidence = ref(0)
const lastSources = ref<string[]>([])

// Web Search 开关（会话级，默认开启）
const webSearchEnabled = ref(true)

// ── 回答模式标签 ──────────────────────────────────────────────
const answerModeTagType = computed(() => {
  switch (lastAnswerMode.value) {
    case 'rag':           return 'success'   // 绿色
    case 'web_augmented': return 'primary'   // 蓝色
    case 'general':       return 'warning'   // 橙色
    default:              return 'info'      // 灰色 (llm_direct)
  }
})

const answerModeLabel = computed(() => {
  switch (lastAnswerMode.value) {
    case 'rag':           return `RAG ${lastConfidence.value}%`
    case 'web_augmented': return `🌐 Web增强`
    case 'general':       return '通用回答'
    default:              return 'LLM直答'
  }
})

// ── 会话管理 ──────────────────────────────────────────────────

function newSession() {
  const id = `student_session_${uuidv4()}`
  const session: Session = { id, name: `会话 ${sessions.value.length + 1}`, messages: [] }
  sessions.value.unshift(session)
  switchSession(id)
}

function switchSession(id: string) {
  if (isStreaming.value) return
  currentSessionId.value = id
  const s = sessions.value.find(s => s.id === id)
  // 直接引用 session.messages，后续只需操作 messages.value，不再单独操作 session.messages
  messages.value = s ? s.messages : []
  scrollToBottom()
}

function deleteSession(id: string) {
  const idx = sessions.value.findIndex(s => s.id === id)
  if (idx === -1) return
  sessions.value.splice(idx, 1)
  if (currentSessionId.value === id) {
    sessions.value.length > 0 ? switchSession(sessions.value[0].id) : newSession()
  }
}

// ── 键盘处理：Enter 发送，Shift+Enter 换行 ────────────────────

function handleKeydown(e: KeyboardEvent) {
  // IME 输入法合成期间（isComposing）不触发发送
  if (e.isComposing) return
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

// ── 发送消息（直接用 fetch 读 ReadableStream，绕过 Vite proxy 缓冲）──

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || isStreaming.value) return

  if (!currentSessionId.value) newSession()

  // 立即清空输入框，防止用户重复触发
  inputText.value = ''

  const session = sessions.value.find(s => s.id === currentSessionId.value)

  // 追加用户消息（messages.value 就是 session.messages 的引用，只 push 一次）
  const userMsg: Message = { role: 'user', content: text }
  messages.value.push(userMsg)
  if (session && session.messages.filter(m => m.role === 'user').length === 1) {
    session.name = text.slice(0, 20) + (text.length > 20 ? '…' : '')
  }

  await scrollToBottom()

  // 开始流式请求
  isStreaming.value = true
  streamingText.value = ''
  progressStage.value = ''
  lastAnswerMode.value = ''
  lastSources.value = []

  try {
    // 直接请求后端，不经过 Vite proxy（避免 proxy 缓冲 SSE 帧）
    const resp = await fetch(`${API_BASE}/api/v1/qa/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth.token}`,
      },
      body: JSON.stringify({
        session_id:        currentSessionId.value,
        message:           text,
        enable_web_search: webSearchEnabled.value,
      }),
    })

    if (resp.status === 401) {
      localStorage.removeItem('edu-agent-token')
      localStorage.removeItem('edu-agent-user')
      router.push('/login')
      return
    }
    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`)
    }

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
          if (evt.type === 'progress' && evt.stage) {
            progressStage.value = evt.stage
          } else if (evt.type === 'token' && evt.content) {
            progressStage.value = ''   // 第一个 token 到达，清除进度提示
            streamingText.value += evt.content
            await scrollToBottom()
          } else if (evt.type === 'meta') {
            lastAnswerMode.value = evt.answer_mode ?? ''
            lastConfidence.value = Math.round((evt.confidence ?? 0) * 100)
            lastSources.value = evt.sources ?? []
          } else if (evt.type === 'error') {
            throw new Error(evt.message ?? 'SSE error')
          }
        } catch (parseErr) {
          // 忽略非 JSON 行
        }
      }
    }

    // 流结束，把流式内容固化为消息（只 push 一次，session.messages 就是 messages.value）
    if (streamingText.value) {
      const assistantMsg: Message = {
        role: 'assistant',
        content: streamingText.value,
        sources: lastSources.value,
      }
      messages.value.push(assistantMsg)
    }

  } catch (err) {
    ElMessage.error('请求失败，请重试')
    console.error('[QA SSE]', err)
  } finally {
    isStreaming.value = false
    streamingText.value = ''
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

onMounted(() => {
  newSession()
})
</script>

<style scoped>
.qa-page {
  display: flex;
  height: calc(100vh - 56px - 48px);
  gap: 16px;
}

/* ── 左侧会话列表 ── */
.session-panel {
  width: 200px;
  background: #fff;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.session-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
  font-size: 14px;
  font-weight: 500;
}
.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}
.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  color: #595959;
  transition: background 0.15s;
}
.session-item:hover { background: #f5f7fa; }
.session-item.active { background: #e6f4ff; color: #1677ff; }
.session-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.delete-btn {
  opacity: 0;
  flex-shrink: 0;
  color: #bfbfbf;
  transition: opacity 0.15s, color 0.15s;
}
.session-item:hover .delete-btn { opacity: 1; }
.delete-btn:hover { color: #ff4d4f !important; }

/* ── 右侧聊天区 ── */
.chat-panel {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.empty-hint {
  text-align: center;
  color: #8c8c8c;
  margin-top: 80px;
  font-size: 15px;
  line-height: 2;
}
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

/* ── 思考中跳点动画 ── */
.thinking {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 0;
}
.progress-hint {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  font-size: 13px;
  color: #8c8c8c;
}
.dot {
  width: 7px;
  height: 7px;
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
