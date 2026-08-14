import client from './client'

export interface ExamSubmitResponse {
  submission_id: string
  status: string
  message: string
}

export interface PendingReviewItem {
  submission_id: string
  student_name: string
  exam_title: string
  submitted_at: string
  pre_review: {
    total_score: number
    full_score: number
    needs_review_count: number
  }
  weak_points: Array<{ tag: string; wrong_count: number; total_count?: number; question_nos?: number[]; suggestion?: string }>
}

export interface ReviewDetail {
  submission_id: string
  student_id: string
  pre_review_summary: {
    total_score: number
    full_score: number
    by_question: Array<{
      question_id: string
      question_no: number
      question_type: string
      full_score: number
      score: number
      content?: string
      student_answer: string
      correct_answer?: string
      ai_feedback: string
      needs_review: boolean
      point_results?: Array<{
        point_score: number
        point_desc: string
        earned: boolean
        missing?: string
      }>
      test_cases_passed?: number
      test_cases_total?: number
      sandbox_skipped?: boolean
      quality_feedback?: string[]
      teacher_comment?: string
      final_score?: number
    }>
  }
  weak_points: Array<{ tag: string; wrong_count: number; total_count?: number; question_nos?: number[]; suggestion?: string }>
  weak_points_summary: string
}

export interface ConfirmRequest {
  action: 'approve' | 'modify'
  modifications: Array<{
    question_id: string
    new_score?: number
    comment?: string
  }>
}

export interface ConfirmResponse {
  submission_id: string
  status: string
  final_score: number
  full_score: number
  score_rate: number
  weak_points: Array<{ tag: string; wrong_count: number; total_count?: number; question_nos?: number[]; suggestion?: string }>
  weak_points_summary: string
}

export interface MySubmissionItem {
  submission_id: string
  exam_id: string
  exam_title: string
  status: string
  submitted_at: string
}

export const examApi = {
  listMySubmissions: () =>
    client.get<{ items: MySubmissionItem[] }>('/exam/my-submissions'),

  submit: (examId: string, file: File) => {
    const form = new FormData()
    form.append('exam_id', examId)
    form.append('file', file)
    return client.post<ExamSubmitResponse>('/exam/submit', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  getPendingReviews: () =>
    client.get<{ items: PendingReviewItem[]; total: number }>('/exam/pending-reviews'),

  getSubmissionReview: (submissionId: string) =>
    client.get<ReviewDetail>(`/exam/my-submissions/${submissionId}`),

  getSubmissionReviewTeacher: (submissionId: string) =>
    client.get<ReviewDetail>(`/exam/submissions/${submissionId}/review`),

  confirmReview: (submissionId: string, data: ConfirmRequest) =>
    client.post<ConfirmResponse>(`/exam/submissions/${submissionId}/confirm`, data),
}
