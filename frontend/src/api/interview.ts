import client from './client'

export interface StartSessionRequest {
  target_position: string
  resume_review_id?: string | null
}

export interface StartSessionResponse {
  session_id: string
  target_position: string
  status: string
  message: string
}

export interface ChatRequest {
  message: string
}

export interface ChatResponse {
  session_id: string
  reply: string
  current_stage: string
  total_turns: number
  is_finished: boolean
  report_summary?: {
    overall_score: number
    strengths: string[]
    improvements: string[]
  }
}

export interface ChatDonePayload {
  type: 'done'
  reply: string          // AI 本轮回复全文（generate_response_node 用 ainvoke，token 不流式）
  current_stage: string
  total_turns: number
  is_finished: boolean
  report_summary?: {
    overall_score: number
    strengths: string[]
    improvements: string[]
  }
}

export interface ChatStreamCallbacks {
  onToken: (chunk: string) => void
  onDone: (payload: ChatDonePayload) => void
  onError: (err: Error) => void
}

export interface DimensionEval {
  dimension: string
  score: number
  comment: string
}

export interface InterviewReport {
  session_id: string
  target_position: string
  overall_score: number
  dimensions: DimensionEval[]
  strengths: string[]
  improvements: string[]
  overall_comment: string
  recommended_topics: string[]
  next_step_advice: string
}

export interface SessionListItem {
  session_id: string
  target_position: string
  overall_score?: number
  status: string
  finished_at?: string
  created_at: string
}

export const interviewApi = {
  startSession: (data: StartSessionRequest) =>
    client.post<StartSessionResponse>('/interview/sessions', data),

  chat: (sessionId: string, data: ChatRequest) =>
    client.post<ChatResponse>(`/interview/sessions/${sessionId}/chat`, data),

  /**
   * 流式对话（SSE）。
   * 使用 fetch + ReadableStream 消费，支持实时 token 渲染。
   * onToken 在每个 token 到达时触发，onDone 在流结束时附带最终状态。
   */
  chatStream(sessionId: string, message: string, callbacks: ChatStreamCallbacks): void {
    const token = localStorage.getItem('edu-agent-token') || ''
    // 直接请求后端，不经过 Vite proxy（避免 proxy 缓冲 SSE 帧，同 QAChatView）
    const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'
    // 标记 done/error 事件是否已分发，用于流结束时的兜底处理
    let doneCalled = false

    const dispatch = (line: string) => {
      if (!line.startsWith('data: ')) return
      try {
        const payload = JSON.parse(line.slice(6))
        if (payload.type === 'token') {
          callbacks.onToken(payload.content)
        } else if (payload.type === 'done') {
          doneCalled = true
          callbacks.onDone(payload as ChatDonePayload)
        } else if (payload.type === 'error') {
          doneCalled = true
          callbacks.onError(new Error(payload.message ?? '流式输出异常'))
        }
      } catch {
        // 忽略心跳、空行等非 JSON 内容
      }
    }

    fetch(`${apiBase}/api/v1/interview/sessions/${sessionId}/chat/stream`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    })
      .then(async (resp) => {
        if (!resp.ok || !resp.body) {
          callbacks.onError(new Error(`HTTP ${resp.status}`))
          return
        }

        const reader = resp.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          // SSE 每条消息以 "\n\n" 结尾，逐段解析
          const parts = buffer.split('\n\n')
          buffer = parts.pop() ?? ''   // 末尾不完整的部分留到下次

          for (const part of parts) {
            for (const line of part.split('\n')) {
              dispatch(line)
            }
          }
        }

        // 流关闭后处理 buffer 中可能残留的最后一帧
        // （极少数情况：最后一个 chunk 末尾没有完整 \n\n）
        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            dispatch(line)
          }
        }

        // 兜底：若连接关闭时 done/error 均未触发，主动通知前端解除等待
        if (!doneCalled) {
          callbacks.onError(new Error('连接已断开，未收到完整响应'))
        }
      })
      .catch((err) => {
        if (!doneCalled) callbacks.onError(err)
      })
  },

  getReport: (sessionId: string) =>
    client.get<InterviewReport>(`/interview/sessions/${sessionId}/report`),

  listSessions: () =>
    client.get<{ items: SessionListItem[]; total: number }>('/interview/sessions'),
}
