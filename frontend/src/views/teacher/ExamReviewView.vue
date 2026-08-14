<template>
  <div class="exam-review">
    <el-card>
      <template #header>
        <span>待确认批改列表</span>
        <el-button style="float: right" :icon="Refresh" circle size="small" @click="fetchList" />
      </template>

      <el-table :data="list" v-loading="loading" size="default">
        <el-table-column prop="student_name" label="学员" width="120" />
        <el-table-column prop="exam_title" label="试卷" />
        <el-table-column prop="submitted_at" label="提交时间" width="180" />
        <el-table-column label="AI 预评分" width="120">
          <template #default="{ row }">
            {{ row.pre_review?.total_score }} / {{ row.pre_review?.full_score }}
          </template>
        </el-table-column>
        <el-table-column label="待确认题数" width="100">
          <template #default="{ row }">
            <el-tag type="warning" size="small">{{ row.pre_review?.needs_review_count }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button type="primary" text size="small" @click="openReview(row.submission_id)">
              审阅
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 审阅抽屉 -->
    <el-drawer v-model="drawerVisible" title="批改确认" size="65%" direction="rtl" destroy-on-close>
      <div v-if="currentReview" class="review-drawer">

        <!-- 汇总信息 -->
        <el-descriptions :column="3" border size="small" style="margin-bottom: 16px">
          <el-descriptions-item label="AI 预评分">
            <b>{{ currentReview.pre_review_summary?.total_score }}</b>
            / {{ currentReview.pre_review_summary?.full_score }} 分
          </el-descriptions-item>
          <el-descriptions-item label="待确认题数">
            <el-tag type="warning" size="small">
              {{ currentReview.pre_review_summary?.by_question?.filter(q => q.needs_review).length }} 题
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="知识薄弱点">
            <span v-if="!currentReview.weak_points?.length" style="color:#999">—</span>
            <el-tag
              v-for="wp in currentReview.weak_points?.slice(0, 3)"
              :key="wp.tag"
              type="danger"
              size="small"
              style="margin-right: 4px"
            >
              {{ wp.tag }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 逐题列表 -->
        <el-collapse v-model="expandedKeys">
          <el-collapse-item
            v-for="q in currentReview.pre_review_summary?.by_question"
            :key="q.question_id"
            :name="q.question_id"
          >
            <template #title>
              <div class="q-title-row">
                <span class="q-no">第 {{ q.question_no }} 题</span>
                <el-tag :type="typeTagType(q.question_type)" size="small" style="margin-left: 6px">
                  {{ typeLabel(q.question_type) }}
                </el-tag>
                <el-tag
                  :type="scoreChanged(q) ? 'danger' : (q.needs_review ? 'warning' : 'success')"
                  size="small"
                  style="margin-left: 6px"
                >
                  {{ scoreChanged(q) ? '已改分 ' + modifications[q.question_id]?.new_score : q.score }}
                  / {{ q.full_score }} 分
                  <span v-if="q.needs_review && !scoreChanged(q)">· 需确认</span>
                </el-tag>
              </div>
            </template>

            <div class="q-detail">
              <!-- 题目内容 -->
              <div v-if="q.content" class="q-section">
                <div class="q-label">题目</div>
                <div class="q-content">{{ q.content }}</div>
              </div>

              <!-- 学员答案 & 参考答案 -->
              <div class="q-row">
                <div class="q-col">
                  <div class="q-label">学员答案</div>
                  <div class="q-answer">{{ q.student_answer || '（未作答）' }}</div>
                </div>
                <div v-if="q.correct_answer" class="q-col">
                  <div class="q-label">参考答案</div>
                  <div class="q-answer correct">{{ q.correct_answer }}</div>
                </div>
              </div>

              <!-- AI 反馈 -->
              <div class="q-section">
                <div class="q-label">AI 批改反馈</div>
                <div class="q-feedback">{{ q.ai_feedback }}</div>
              </div>

              <!-- 得分点（简答题） -->
              <div v-if="q.point_results?.length" class="q-section">
                <div class="q-label">得分点明细</div>
                <div v-for="(pt, i) in q.point_results" :key="i" class="point-row">
                  <el-icon :color="pt.earned ? '#67c23a' : '#f56c6c'">
                    <component :is="pt.earned ? 'CircleCheck' : 'CircleClose'" />
                  </el-icon>
                  <span style="margin-left: 4px">{{ pt.point_desc }}（{{ pt.point_score }}分）</span>
                  <span v-if="!pt.earned && pt.missing" style="color:#f56c6c; margin-left: 4px">— {{ pt.missing }}</span>
                </div>
              </div>

              <!-- 代码题：测试用例 -->
              <div v-if="q.question_type === 'code'" class="q-section">
                <div class="q-label">测试用例</div>
                <span v-if="q.sandbox_skipped" style="color:#909399">Judge0 沙箱跳过</span>
                <span v-else>通过 {{ q.test_cases_passed }} / {{ q.test_cases_total }}</span>
              </div>

              <!-- 教师改分区 -->
              <div class="q-modify-row">
                <span class="q-label" style="min-width: 64px">教师改分</span>
                <el-input-number
                  v-model="modifications[q.question_id].new_score"
                  :min="0"
                  :max="q.full_score"
                  size="small"
                  style="width: 110px"
                  @change="onScoreChange(q)"
                />
                <span style="font-size:12px; color:#909399; margin: 0 4px">/ {{ q.full_score }}</span>
                <el-input
                  v-model="modifications[q.question_id].comment"
                  placeholder="教师批注（可选）"
                  size="small"
                  style="flex: 1; min-width: 180px; max-width: 320px"
                />
              </div>
            </div>
          </el-collapse-item>
        </el-collapse>

        <!-- 操作栏 -->
        <div class="action-bar">
          <div style="font-size:13px; color:#606266">
            <span v-if="changedCount === 0">AI 批改结果未做修改，点击确认直接发布。</span>
            <span v-else>已修改 <b>{{ changedCount }}</b> 题分数，确认后以修改值为准发布。</span>
          </div>
          <div style="display:flex; gap:10px; margin-top:10px">
            <el-button @click="drawerVisible = false" :disabled="confirming">取消</el-button>
            <el-button type="primary" :loading="confirming" @click="confirmReview">
              {{ changedCount > 0 ? '修改后确认发布' : '确认发布' }}
            </el-button>
          </div>
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, CircleCheck, CircleClose } from '@element-plus/icons-vue'
import { examApi, type PendingReviewItem, type ReviewDetail } from '@/api/exam'

const loading = ref(false)
const list = ref<PendingReviewItem[]>([])
const drawerVisible = ref(false)
const currentReview = ref<ReviewDetail | null>(null)
const confirming = ref(false)
const expandedKeys = ref<string[]>([])

// 每题：{ new_score, comment }，初始化为 AI 分数
const modifications = reactive<Record<string, { new_score: number; comment: string }>>({})

// 计算被修改的题目数（和 AI 分数不同的）
const changedCount = computed(() => {
  if (!currentReview.value) return 0
  return currentReview.value.pre_review_summary.by_question.filter(q => scoreChanged(q)).length
})

function scoreChanged(q: { question_id: string; score: number }) {
  const mod = modifications[q.question_id]
  return mod !== undefined && mod.new_score !== q.score
}

function onScoreChange(q: { question_id: string; score: number }) {
  // 触发 changedCount 重计算（reactive 已自动处理，此处留作扩展点）
}

function typeLabel(type: string) {
  const map: Record<string, string> = {
    single_choice: '单选',
    multi_choice:  '多选',
    judge:         '判断',
    short_answer:  '简答',
    code:          '代码',
  }
  return map[type] ?? type
}

function typeTagType(type: string) {
  const map: Record<string, string> = {
    single_choice: '',
    multi_choice:  '',
    judge:         '',
    short_answer:  'warning',
    code:          'danger',
  }
  return (map[type] ?? '') as '' | 'success' | 'warning' | 'danger' | 'info'
}

async function fetchList() {
  loading.value = true
  try {
    const { data } = await examApi.getPendingReviews()
    list.value = data.items
  } finally {
    loading.value = false
  }
}

async function openReview(submissionId: string) {
  const { data } = await examApi.getSubmissionReviewTeacher(submissionId)
  currentReview.value = data

  // 初始化改分表：默认使用 AI 分数
  for (const q of data.pre_review_summary?.by_question ?? []) {
    modifications[q.question_id] = { new_score: q.score, comment: '' }
  }

  // 自动展开需要确认的题目
  expandedKeys.value = (data.pre_review_summary?.by_question ?? [])
    .filter(q => q.needs_review)
    .map(q => q.question_id)

  drawerVisible.value = true
}

async function confirmReview() {
  if (!currentReview.value) return
  confirming.value = true
  try {
    const questions = currentReview.value.pre_review_summary.by_question ?? []

    // 只提交实际改动的题目
    const changedMods = questions
      .filter(q => scoreChanged(q))
      .map(q => ({
        question_id: q.question_id,
        new_score:   modifications[q.question_id].new_score,
        comment:     modifications[q.question_id].comment || undefined,
      }))

    const action = changedMods.length > 0 ? 'modify' : 'approve'

    await examApi.confirmReview(currentReview.value.submission_id, {
      action,
      modifications: changedMods,
    })

    ElMessage.success('批改结果已发布')
    drawerVisible.value = false
    fetchList()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail ?? '发布失败，请重试')
  } finally {
    confirming.value = false
  }
}

onMounted(fetchList)
</script>

<style scoped>
.exam-review { max-width: 1100px; }

.q-title-row {
  display: flex;
  align-items: center;
  gap: 0;
}
.q-no { font-weight: 600; }

.q-detail {
  padding: 4px 0 8px 0;
  font-size: 14px;
  line-height: 1.8;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.q-section { display: flex; flex-direction: column; gap: 2px; }

.q-row {
  display: flex;
  gap: 24px;
}
.q-col { flex: 1; display: flex; flex-direction: column; gap: 2px; }

.q-label {
  font-size: 12px;
  color: #909399;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.q-content {
  background: #f5f7fa;
  border-radius: 4px;
  padding: 6px 10px;
  white-space: pre-wrap;
  word-break: break-all;
}

.q-answer {
  background: #f5f7fa;
  border-radius: 4px;
  padding: 6px 10px;
  white-space: pre-wrap;
  word-break: break-all;
  min-height: 32px;
}
.q-answer.correct {
  background: #f0f9eb;
  color: #529b2e;
}

.q-feedback {
  color: #606266;
  padding: 4px 0;
  white-space: pre-wrap;
}

.point-row {
  display: flex;
  align-items: center;
  font-size: 13px;
  padding: 2px 0;
}

.q-modify-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 8px 10px;
  background: #fafafa;
  border-radius: 4px;
  border: 1px solid #e4e7ed;
}

.action-bar {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #e4e7ed;
}
</style>
