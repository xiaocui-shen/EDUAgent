<template>
  <div class="interview-setup">
    <el-card style="max-width: 560px">
      <template #header><span>🎤 开始模拟面试</span></template>

      <el-form :model="form" label-width="110px">
        <el-form-item label="目标岗位">
          <el-input
            v-model="form.targetPosition"
            placeholder="例如：Java 后端开发工程师"
          />
        </el-form-item>
        <el-form-item label="关联简历（可选）">
          <el-select
            v-model="form.resumeReviewId"
            placeholder="选择已审查的简历（可选）"
            clearable
            style="width: 100%"
          >
            <el-option
              v-for="r in resumeHistory"
              :key="r.review_id"
              :label="`${r.review_id.slice(0, 8)}... (${r.created_at})`"
              :value="r.review_id"
            />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="loading"
            :disabled="!form.targetPosition.trim()"
            @click="startInterview"
          >
            开始面试
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 历史面试记录 -->
    <el-card style="margin-top: 16px; max-width: 800px">
      <template #header>
        <span>历史面试记录</span>
        <el-button style="float: right" :icon="Refresh" circle size="small" @click="fetchHistory" />
      </template>
      <el-empty v-if="!history.length" description="暂无记录" />
      <el-table v-else :data="history" size="small">
        <el-table-column prop="target_position" label="目标岗位" />
        <el-table-column label="综合评分" width="100">
          <template #default="{ row }">
            <span v-if="row.overall_score">{{ row.overall_score }}</span>
            <el-tag v-else type="info" size="small">进行中</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <el-tag :type="row.status === 'finished' ? 'success' : 'warning'" size="small">
              {{ row.status === 'finished' ? '已完成' : '进行中' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="时间" width="180" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button
              text type="primary" size="small"
              @click="router.push(`/interview/${row.session_id}`)"
            >
              {{ row.status === 'finished' ? '查看报告' : '继续' }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Refresh } from '@element-plus/icons-vue'
import { interviewApi, type SessionListItem } from '@/api/interview'
import { resumeApi, type ReviewListItem } from '@/api/resume'

const router = useRouter()
const loading = ref(false)
const form = reactive({ targetPosition: '', resumeReviewId: '' })
const history = ref<SessionListItem[]>([])
const resumeHistory = ref<ReviewListItem[]>([])

async function startInterview() {
  if (!form.targetPosition.trim()) return
  loading.value = true
  try {
    const { data } = await interviewApi.startSession({
      target_position: form.targetPosition,
      resume_review_id: form.resumeReviewId || null,
    })
    router.push({ path: `/interview/${data.session_id}`, state: { openingMessage: data.message } })
  } finally {
    loading.value = false
  }
}

async function fetchHistory() {
  try {
    const { data } = await interviewApi.listSessions()
    history.value = data.items
  } catch { /* ignore */ }
}

onMounted(async () => {
  fetchHistory()
  try {
    const { data } = await resumeApi.listReviews()
    resumeHistory.value = data.items.filter(r => r.status === 'done')
  } catch { /* ignore */ }
})
</script>

<style scoped>
.interview-setup { max-width: 800px; }
</style>
