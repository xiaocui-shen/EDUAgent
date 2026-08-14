import client from './client'

export interface ChatRequest {
  session_id: string
  course_id?: string | null
  message: string
}

export interface ChatResponse {
  session_id: string
  answer: string
  answer_mode: 'rag' | 'llm_direct'
  confidence: number
  sources: string[]
  fallback_used: boolean
}

export interface HistoryResponse {
  session_id: string
  messages: Array<{ role: string; content: string; created_at: string }>
  summary: string | null
  total_turns: number
}

export const qaApi = {
  chat: (data: ChatRequest) =>
    client.post<ChatResponse>('/qa/chat', data),

  getHistory: (sessionId: string) =>
    client.get<HistoryResponse>(`/qa/sessions/${sessionId}/history`),
}
