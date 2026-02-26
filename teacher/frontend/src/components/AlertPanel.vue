<template>
  <div class="alert-panel" v-if="alerts.length > 0">
    <div class="panel-header" @click="expanded = !expanded">
      <span class="panel-title">
        Risk Alerts
        <span class="alert-count">{{ alerts.length }}</span>
      </span>
      <span class="toggle-icon">{{ expanded ? '&#9660;' : '&#9654;' }}</span>
    </div>

    <transition name="slide">
      <div v-if="expanded" class="panel-body">
        <div
          v-for="alert in alerts"
          :key="alert.station_id"
          class="alert-item"
          :class="'alert-' + alert.risk_level"
        >
          <div class="alert-main">
            <span class="badge" :class="'badge-' + alert.risk_level">
              {{ alert.risk_level }}
            </span>
            <router-link
              :to="{ name: 'station', params: { id: alert.station_id } }"
              class="alert-station"
            >
              {{ alert.station_id }}
            </router-link>
            <span class="alert-student" v-if="alert.student_name">
              {{ alert.student_name }}
            </span>
          </div>
          <div class="alert-reasons">
            <span v-for="(reason, i) in alert.risk_reasons" :key="i" class="reason-tag">
              {{ reason }}
            </span>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useClassroomStore } from '../stores/classroom.js'

const store = useClassroomStore()
const expanded = ref(true)

const alerts = computed(() => store.alerts || [])
</script>

<style scoped>
.alert-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
  transition: background var(--transition);
}
.panel-header:hover {
  background: rgba(255, 255, 255, 0.03);
}
.panel-title {
  font-weight: 600;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.alert-count {
  background: var(--danger);
  color: #fff;
  font-size: 11px;
  padding: 1px 7px;
  border-radius: 10px;
  font-weight: 700;
}
.toggle-icon {
  color: var(--text-muted);
  font-size: 12px;
}

.panel-body {
  border-top: 1px solid var(--border);
  max-height: 300px;
  overflow-y: auto;
}
.alert-item {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  transition: background var(--transition);
}
.alert-item:last-child {
  border-bottom: none;
}
.alert-item:hover {
  background: rgba(255, 255, 255, 0.02);
}
.alert-item.alert-danger {
  border-left: 3px solid var(--danger);
}
.alert-item.alert-warning {
  border-left: 3px solid var(--warning);
}

.alert-main {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.alert-station {
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
}
.alert-station:hover {
  text-decoration: underline;
}
.alert-student {
  color: var(--text-muted);
  font-size: 13px;
}

.alert-reasons {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.reason-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  background: rgba(255, 85, 85, 0.1);
  color: var(--text-secondary);
}

.slide-enter-active,
.slide-leave-active {
  transition: max-height 0.3s ease, opacity 0.3s ease;
  overflow: hidden;
}
.slide-enter-from,
.slide-leave-to {
  max-height: 0;
  opacity: 0;
}
</style>
