<template>
  <nav class="navbar">
    <div class="navbar-inner">
      <div class="nav-brand">
        <span class="brand-icon">&#9879;</span>
        <span class="brand-text">LabGuardian</span>
        <span class="brand-sub">Teacher</span>
      </div>

      <div class="nav-links">
        <router-link to="/" class="nav-link" exact-active-class="active">
          Overview
        </router-link>
        <router-link to="/ranking" class="nav-link" active-class="active">
          Ranking
        </router-link>
        <router-link to="/stats" class="nav-link" active-class="active">
          Statistics
        </router-link>
      </div>

      <div class="nav-status">
        <span class="status-dot" :class="onlineCount > 0 ? 'online' : 'offline'"></span>
        <span class="status-text">{{ onlineCount }} / {{ totalCount }} online</span>
        <span v-if="dangerCount > 0" class="danger-count">{{ dangerCount }} alert</span>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { computed } from 'vue'
import { useClassroomStore } from '../stores/classroom.js'

const store = useClassroomStore()

const onlineCount = computed(() => store.onlineCount)
const totalCount = computed(() => store.totalCount)
const dangerCount = computed(() => store.dangerStations.length)
</script>

<style scoped>
.navbar {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  height: 52px;
  position: sticky;
  top: 0;
  z-index: 100;
}
.navbar-inner {
  max-width: 1440px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  height: 100%;
  gap: 32px;
}
.nav-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 16px;
  flex-shrink: 0;
}
.brand-icon {
  font-size: 22px;
  color: var(--accent);
}
.brand-text {
  color: var(--text-primary);
}
.brand-sub {
  color: var(--text-muted);
  font-weight: 400;
  font-size: 13px;
}
.nav-links {
  display: flex;
  gap: 4px;
}
.nav-link {
  padding: 6px 14px;
  border-radius: var(--radius);
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: all var(--transition);
}
.nav-link:hover {
  color: var(--text-primary);
  background: rgba(108, 99, 255, 0.1);
}
.nav-link.active {
  color: var(--accent);
  background: rgba(108, 99, 255, 0.15);
}
.nav-status {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.status-dot.online {
  background: var(--success);
  box-shadow: 0 0 6px rgba(80, 250, 123, 0.5);
}
.status-dot.offline {
  background: var(--text-muted);
}
.status-text {
  color: var(--text-secondary);
}
.danger-count {
  background: rgba(255, 85, 85, 0.15);
  color: var(--danger);
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 600;
  font-size: 12px;
}

@media (max-width: 768px) {
  .navbar {
    padding: 0 12px;
  }
  .brand-sub {
    display: none;
  }
  .nav-links {
    gap: 2px;
  }
  .nav-link {
    padding: 4px 8px;
    font-size: 13px;
  }
}
</style>
