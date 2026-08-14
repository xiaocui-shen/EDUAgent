<template>
  <div class="interview-chat">
    <!-- 阶段进度条 -->
    <el-card style="margin-bottom: 16px; padding: 0">
      <StageProgressBar :current="currentStage" />
    </el-card>

    <!-- 报告（面试结束后显示） -->
    <template v-if="isFinished && report">
      <el-card class="report-card">
        <template #header>
          <span>🎉 面试报告 — {{ report.target_position }}</span>
        </template>

        <el-row :gutter="16" style="margin-bottom: 16px">
          <el-col :span="6" style="text-align: center">
            <div class="overall-score">{{ report.overall_score }}</div>
            <div style="color: #8c8c8c; font-size: 13px">综合评分（满分 100）</div>
          </el-col>
          <el-col :span="18">
            <div class="report-comment">{{ report.overall_comment }}</div>
          </el-col>
        </el-row>

        <el-row :gutter="16" style="margin-bottom: 16px">
          <el-col :span="12">
            <div class="report-section-title">✅ 优势</div>
            <ul class="report-list">
              <li v-for="(s, i) in report.strengths" :key="i">{{ s }}</li>
            </ul>
          </el-col>
          <el-col :span="12">
            <div class="report-section-title">🔧 待提升</div>
            <ul class="report-list">
              <li v-for="(s, i) in report.improvements" :key="i">{{ s }}</li>
            </ul>
          </el-col>
        </el-row>

        <el-row :gutter="16">
          <el-col :span="12">
            <div class="report-section-title">📚 建议学习方向</div>
            <el-tag
              v-for="(t, i) in report.recommended_topics"
              :key="i"
              style="margin: 4px"
            >{{ t }}</el-tag>
          </el-col>
          <el-col :span="12">
            <div class="report-section-title">🚀 下一步建议</div>
            <div style="font-size: 14px; color: #595959">{{ report.next_step_advice }}</div>
          </el-col>
        </el-row>

        <el-divider />
        <div class="report-section-title" style="margin-bottom: 12px">各维度评分</div>
        <el-row :gutter="12">
          <el-col :span="8" v-for="dim in report.dimensions" :key="dim.dimension" style="margin-bottom: 12px">
            <el-card shadow="never" style="border: 1px solid #f0f0f0">
              <div style="display: flex; justify-content: space-between; align-items: center">
                <span style="font-size: 13px">{{ dim.dimension }}</span>
                <span style="font-size: 18px; font-weight: 700; color: #1677ff">{{ dim.score }}</span>
              </div>
              <el-progress
                :percentage="dim.score"
                :show-text="false"
                :stroke-width="6"
                style="margin-top: 6px"
              />
              <div style="font-size: 12px; color: #8c8c8c; margin-top: 4px">{{ dim.comment }}</div>
            </el-card>
          </el-col>
        </el-row>

        <div style="margin-top: 16px">
          <el-button @click="router.push('/interview')">返回列表</el-button>
        </div>
      </el-card>
    </template>

    <!-- 聊天区（面试进行中） -->
    <template v-else>
      <el-card class="chat-card">
        <div class="chat-messages" ref="messagesEl">
          <ChatBubble
            v-for="(msg, i) in messages"
            :key="i"
            :role="msg.role"
          >
            <MarkdownRenderer :content="msg.content" />
          </ChatBubble>
          <!-- 流式输出时显示光标动画，同步等待时显示"思考中" -->
          <ChatBubble v-if="loading && !streamingReply" role="assistant">
            <span style="color: #8c8c8c">思考中...</span>
          </ChatBubble>
        </div>

        <div class="chat-input-area">
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="3"
            placeholder="输入你的回答，Ctrl+Enter 发送"
            resize="none"
            :disabled="loading || isFinished"
            @keydown.ctrl.enter="sendMessage"
          />

          <div class="input-actions">
            <!-- 左侧：轮次 -->
            <div style="display: flex; align-items: center; gap: 12px">
              <span style="font-size: 12px; color: #8c8c8c">第 {{ totalTurns }} 轮</span>
            </div>

            <!-- 右侧：结束 + 发送 -->
            <div style="display: flex; gap: 8px; align-items: center">
              <el-button
                type="danger"
                plain
                size="small"
                :disabled="loading || isFinished"
                @click="endInterview"
              >
                结束面试
              </el-button>
              <el-button
                type="primary"
                :loading="loading"
                :disabled="!inputText.trim() || isFinished"
                @click="sendMessage"
              >
                发送
              </el-button>
            </div>
          </div>
        </div>
      </el-card>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import ChatBubble from '@/components/chat/ChatBubble.vue'
import MarkdownRenderer from '@/components/chat/MarkdownRenderer.vue'
import StageProgressBar from '@/components/interview/StageProgressBar.vue'
import { interviewApi, type InterviewReport } from '@/api/interview'
import client from '@/api/client'

interface Message { role: 'user' | 'assistant'; content: string }

const route = useRoute()
const router = useRouter()
const sessionId = route.params.sessionId as string

// ── 聊天状态 ──────────────────────────────────────────────────
const messages = ref<Message[]>([])
const inputText = ref('')
const loading = ref(false)
const streamingReply = ref(false)  // 流式输出进行中（区别于初始等待）
const currentStage = ref('warmup')
const totalTurns = ref(0)
const isFinished = ref(false)
const report = ref<InterviewReport | null>(null)
const messagesEl = ref<HTMLElement>()


// ── 发送消息（流式） ──────────────────────────────────────────
async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  loading.value = true
  await scrollToBottom()

  // 不预插入气泡：等第一个 token 或 done 事件到来时再插入，
  // 避免与"思考中..."条件气泡同时出现导致双气泡。
  let streamBubbleIndex = -1  // -1 表示尚未插入 assistant 气泡

  return new Promise<void>((resolve) => {
    interviewApi.chatStream(sessionId, text, {
      onToken: async (chunk) => {
        // 第一个 token 到来时才插入气泡
        if (streamBubbleIndex === -1) {
          messages.value.push({ role: 'assistant', content: '' })
          streamBubbleIndex = messages.value.length - 1
          streamingReply.value = true
        }
        messages.value[streamBubbleIndex].content += chunk
        await scrollToBottom()
      },
      onDone: async (meta) => {
        streamingReply.value = false
        currentStage.value = meta.current_stage
        totalTurns.value = meta.total_turns

        if (streamBubbleIndex === -1) {
          // generate_response_node 使用 ainvoke（无 token 事件），
          // 回复内容由 done 事件的 reply 字段携带，此时才插入气泡。
          if (meta.reply) {
            messages.value.push({ role: 'assistant', content: meta.reply })
            streamBubbleIndex = messages.value.length - 1
          }
          // 若 reply 也为空（generate_report 路径），则不插入气泡
        } else if (!messages.value[streamBubbleIndex]?.content && meta.reply) {
          // 有气泡但内容为空（极少情况）：用 reply 填充
          messages.value[streamBubbleIndex].content = meta.reply
        }

        const replyContent = messages.value[streamBubbleIndex]?.content ?? ''

        if (meta.is_finished) {
          isFinished.value = true
          await fetchReport()
        }

        loading.value = false
        await scrollToBottom()
        resolve()
      },
      onError: async (_err) => {
        streamingReply.value = false
        // 若已插入气泡但内容为空，移除避免残留
        if (streamBubbleIndex !== -1 && !messages.value[streamBubbleIndex]?.content) {
          messages.value.splice(streamBubbleIndex, 1)
        }
        ElMessage.error('发送失败，请重试')
        loading.value = false
        await scrollToBottom()
        resolve()
      },
    })
  })
}

async function endInterview() {
  inputText.value = '结束面试'
  await sendMessage()
}

async function fetchReport() {
  try {
    const res = await client.get(`/interview/sessions/${sessionId}/report`, {
      validateStatus: (s: number) => s < 500,
    })
    if (res.status === 200) report.value = res.data
  } catch { /* ignore */ }
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

// ── 生命周期 ──────────────────────────────────────────────────
onMounted(async () => {
  const openingMessage = (history.state as any)?.openingMessage
  if (openingMessage) {
    messages.value.push({ role: 'assistant', content: openingMessage })
  }

  try {
    const res = await client.get(`/interview/sessions/${sessionId}/report`, {
      validateStatus: (s: number) => s < 500,
    })
    if (res.status === 200 && res.data?.overall_score) {
      isFinished.value = true
      report.value = res.data
      currentStage.value = 'finished'
    }
  } catch { /* ignore */ }
})

</script>

<style scoped>
.interview-chat { max-width: 900px; }
.chat-card { display: flex; flex-direction: column; }
.chat-messages {
  height: calc(100vh - 56px - 48px - 120px - 200px);
  overflow-y: auto;
  padding: 8px 0;
}
.chat-input-area { border-top: 1px solid #f0f0f0; padding-top: 12px; margin-top: 12px; }
.input-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
}
.overall-score { font-size: 52px; font-weight: 700; color: #1677ff; line-height: 1; }
.report-comment { font-size: 14px; color: #595959; line-height: 1.8; }
.report-section-title { font-weight: 500; margin-bottom: 8px; }
.report-list { padding-left: 20px; margin: 0; font-size: 14px; line-height: 1.8; }
.report-card { margin-bottom: 16px; }


</style>
