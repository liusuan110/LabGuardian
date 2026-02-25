<template>
  <router-link
    :to="{ name: 'station', params: { id: station.station_id } }"
    class="station-card card"
    :class="{ offline: !station.online, danger: station.risk_level === 'danger' }"
  >
    <div class="card-header">
      <span class="station-id">{{ station.station_id }}</span>
      <span class="badge" :class="'badge-' + riskClass">{{ riskLabel }}</span>
    </div>

    <div class="card-thumb">
      <img
        v-if="station.thumbnail_b64 && station.online"
        :src="'data:image/jpeg;base64,' + station.thumbnail_b64"
        alt="thumbnail"
      />
      <div v-else class="thumb-placeholder">
        <span v-if="!station.online">OFFLINE</span>
        <span v-else>NO IMAGE</span>
      </div>
    </div>

    <div class="card-body">
      <div class="student-name">{{ station.student_name || '--' }}</div>
      <div class="card-metrics">
        <ProgressRing :value="station.progress || 0" :size="44" :stroke-width="3" />
        <div class="metrics-text">
          <div class="metric-row">
            <span class="metric-label">Components</span>
            <span class="metric-value">{{ station.component_count || 0 }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Match</span>
            <span class="metric-value">{{ station.match_level || '--' }}</span>
          </div>
        </div>
      </div>
    </div>

    <div v-if="station.diagnostics && station.diagnostics.length > 0" class="card-diag">
      {{ station.diagnostics[0] }}
    </div>
  </router-link>
</template>

<script setup>
import { computed } from 'vue'
import ProgressRing from './ProgressRing.vue'

const props = defineProps({
  station: { type: Object, required: true },
})

const riskClass = computed(() => {
  if (!props.station.online) return 'offline'
  return props.station.risk_level || 'safe'
})

const riskLabel = computed(() => {
  if (!props.station.online) return 'Offline'
  const map = { safe: 'Safe', warning: 'Warning', danger: 'Danger' }
  return map[props.station.risk_level] || 'Safe'
})
</script>

<style scoped>
.station-card {
  display: flex;
  flex-direction: column;
  text-decoration: none;
  color: inherit;
  cursor: pointer;
  overflow: hidden;
  transition: transform var(--transition), border-color var(--transition);
}
.station-card:hover {
  transform: translateY(-2px);
}
.station-card.offline {
  opacity: 0.55;
}
.station-card.danger {
  border-color: var(--danger);
  box-shadow: 0 0 12px rgba(255, 85, 85, 0.15);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.station-id {
  font-weight: 700;
  font-size: 15px;
  color: var(--text-primary);
}

.card-thumb {
  width: 100%;
  aspect-ratio: 4/3;
  background: var(--bg-primary);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 10px;
}
.card-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.thumb-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 12px;
  letter-spacing: 1px;
}

.card-body {
  flex: 1;
}
.student-name {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.card-metrics {
  display: flex;
  align-items: center;
  gap: 12px;
}
.metrics-text {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.metric-row {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
}
.metric-label {
  color: var(--text-muted);
}
.metric-value {
  color: var(--text-primary);
  font-weight: 600;
}

.card-diag {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--warning);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
