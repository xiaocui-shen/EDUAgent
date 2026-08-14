import { ref } from 'vue'
import { useAuthStore } from '@/stores/auth'

export interface SSEToken {
  type: 'token' | 'meta' | 'done' | 'error'
  content?: string
  answer_mode?: string
  confidence?: number
  sources?: string[]
  fallback_used?: boolean
  message?: string
}

export function useSSEChat() {
  const streamingContent = ref('')
  const isStreaming = ref(false)
  const lastMeta = ref<Omit<SSEToken, 'type' | 'content'> | null>(null)
  const error = ref<string | null>(null)

  async function sendStream(
    sessionId: string,
    message: string,
    courseId?: string | null,
    onToken?: (token: string) => void,
  ) {
    isStreaming.value = true
    streamingContent.value = ''
    lastMeta.value = null
    error.value = null

    const auth = useAuthStore()
    const baseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

    try {
      const response = await fetch(`${baseUrl}/qa/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${auth.token}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          message,
          course_id: courseId ?? null,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue

          try {
            const evt: SSEToken = JSON.parse(raw)
            if (evt.type === 'token' && evt.content) {
              streamingContent.value += evt.content
              onToken?.(evt.content)
            } else if (evt.type === 'meta') {
              lastMeta.value = {
                answer_mode: evt.answer_mode,
                confidence: evt.confidence,
                sources: evt.sources,
                fallback_used: evt.fallback_used,
              }
            } else if (evt.type === 'done') {
              isStreaming.value = false
            } else if (evt.type === 'error') {
              error.value = evt.message ?? '流式响应出错'
              isStreaming.value = false
            }
          } catch {
            // 忽略非 JSON 行
          }
        }
      }
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : '连接失败'
    } finally {
      isStreaming.value = false
    }
  }

  return { streamingContent, isStreaming, lastMeta, error, sendStream }
}
