<template>
  <div class="sidebar">
    <div class="logo">
      <span>🎓 EduAgent</span>
    </div>
    <nav class="nav-list">
      <RouterLink to="/dashboard" class="nav-item" :class="{ 'nav-item--active': isActive('/dashboard') }">
        <el-icon><House /></el-icon>
        <span>首页</span>
      </RouterLink>

      <RouterLink to="/qa" class="nav-item" :class="{ 'nav-item--active': isActive('/qa') }">
        <el-icon><ChatDotRound /></el-icon>
        <span>智能问答</span>
      </RouterLink>

      <RouterLink to="/exam" class="nav-item" :class="{ 'nav-item--active': isActive('/exam') }">
        <el-icon><Document /></el-icon>
        <span>试卷批改</span>
      </RouterLink>

      <RouterLink to="/resume" class="nav-item" :class="{ 'nav-item--active': isActive('/resume') }">
        <el-icon><Postcard /></el-icon>
        <span>简历审查</span>
      </RouterLink>

      <RouterLink to="/interview" class="nav-item" :class="{ 'nav-item--active': isActive('/interview') }">
        <el-icon><Microphone /></el-icon>
        <span>模拟面试</span>
      </RouterLink>

      <!-- 教师端菜单（仅 teacher/admin 可见） -->
      <template v-if="auth.isTeacher">
        <div class="nav-divider" />
        <RouterLink to="/teacher/exam-review" class="nav-item" :class="{ 'nav-item--active': isActive('/teacher/exam-review') }">
          <el-icon><EditPen /></el-icon>
          <span>批改确认</span>
        </RouterLink>
        <RouterLink to="/teacher/knowledge-pending" class="nav-item" :class="{ 'nav-item--active': isActive('/teacher/knowledge-pending') }">
          <el-icon><Collection /></el-icon>
          <span>知识库待补充</span>
        </RouterLink>
      </template>
    </nav>
  </div>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'
import {
  House, ChatDotRound, Document, Postcard,
  Microphone, EditPen, Collection,
} from '@element-plus/icons-vue'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const route = useRoute()

// 仅用于子路由高亮（纯视觉反馈，不参与导航逻辑）
// RouterLink 的 active-class 基于路由记录层级，不覆盖平级子路由（如 /resume/:id）
function isActive(prefix: string) {
  return route.path === prefix || route.path.startsWith(prefix + '/')
}
</script>

<style scoped>
.sidebar {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 16px;
  font-weight: 600;
  border-bottom: 1px solid #ffffff1a;
  flex-shrink: 0;
}

.nav-list {
  display: flex;
  flex-direction: column;
  padding: 4px 0;
  flex: 1;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 13px 20px;
  color: #ffffffa6;
  text-decoration: none;
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.2s, color 0.2s;
  user-select: none;
}

.nav-item:hover {
  background-color: #ffffff14;
  color: #fff;
}

.nav-item--active {
  background-color: #1677ff;
  color: #fff;
}

.nav-item .el-icon {
  font-size: 16px;
  flex-shrink: 0;
}

.nav-divider {
  border: none;
  border-top: 1px solid #ffffff1a;
  margin: 8px 0;
}
</style>
