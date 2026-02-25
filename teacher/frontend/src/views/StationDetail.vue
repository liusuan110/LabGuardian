<template>
  <div class="station-detail">
    <div class="page-header">
      <div>
        <router-link to="/" class="back-link">&larr; Back to overview</router-link>
        <h1>
          Station {{ id }}
          <span v-if="station" class="badge" :class="'badge-' + (station.online ? station.risk_level : 'offline')">
            {{ station.online ? station.risk_level : 'offline' }}
          </span>
        </h1>
        <p v-if="station">{{ station.student_name || 'No student name' }}</p>
      </div>
    </div>

    <div v-if="!station" class="empty-state">
      <div class="icon">&#128269;</div>
      <p>Station not found or not yet connected.</p>
    </div>

    <template v-else>
      <div class="detail-grid">
        <!-- Left: Video Stream -->
        <div class="video-section card">
          <div class="video-header">Live View</div>
          <div class="video-container">
            <img
              v-if="station.online && station.thumbnail_b64"
              :src="'data:image/jpeg;base64,' + station.thumbnail_b64"
              class="video-frame"
              alt="station view"
            />
            <div v-else class="video-placeholder">
              <span v-if="!station.online">Station Offline</span>
              <span v-else>No video feed</span>
            </div>
          </div>
          <div class="video-meta">
            <span>FPS: {{ station.fps?.toFixed(1) || '0.0' }}</span>
            <span>Detector: {{ station.detector_ok ? 'OK' : 'Error' }}</span>
            <span v-if="station.llm_backend">LLM: {{ station.llm_backend }}</span>
          </div>
        </div>

        <!-- Right: Metrics -->
        <div class="metrics-section">
          <!-- Progress -->
          <div class="card metric-card">
            <div class="metric-grid">
              <div class="metric-item">
                <span class="metric-label">Progress</span>
                <div class="progress-bar" style="height: 10px">
                  <div
                    class="fill"
                    :class="{ complete: (station.progress || 0) >= 1.0 }"
                    :style="{ width: Math.round((station.progress || 0) * 100) + '%' }"
                  ></div>
                </div>
                <span class="metric-value-lg">{{ Math.round((station.progress || 0) * 100) }}%</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Similarity</span>
                <span class="metric-value-lg">{{ ((station.similarity || 0) * 100).toFixed(1) }}%</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Match Level</span>
                <span class="metric-value-lg">{{ station.match_level || '--' }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Components</span>
                <span class="metric-value-lg">{{ station.component_count || 0 }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Nets</span>
                <span class="metric-value-lg">{{ station.net_count || 0 }}</span>
              </div>
            </div>
          </div>

          <!-- Missing Components -->
          <div v-if="station.missing_components && station.missing_components.length > 0" class="card">
            <div class="card-section-title">Missing Components</div>
            <div class="tag-list">
              <span v-for="c in station.missing_components" :key="c" class="tag tag-warning">
                {{ c }}
              </span>
            </div>
          </div>

          <!-- Diagnostics -->
          <div v-if="station.diagnostics && station.diagnostics.length > 0" class="card">
            <div class="card-section-title">Diagnostics</div>
            <ul class="diag-list">
              <li
                v-for="(d, i) in station.diagnostics"
                :key="i"
                :class="diagClass(d)"
              >
                {{ d }}
              </li>
            </ul>
          </div>

          <!-- Circuit Snapshot -->
          <div v-if="station.circuit_snapshot" class="card">
            <div class="card-section-title">Circuit Snapshot</div>
            <pre class="snapshot">{{ station.circuit_snapshot }}</pre>
          </div>
        </div>
      </div>

      <!-- Guidance -->
      <div class="guidance-section">
        <GuidanceBox :station-id="id" />
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useClassroomStore } from '../stores/classroom.js'
import GuidanceBox from '../components/GuidanceBox.vue'

const props = defineProps({
  id: { type: String, required: true },
})

const store = useClassroomStore()

const station = computed(() => {
  const s = store.stations[props.id]
  if (!s) return null
  return { station_id: props.id, ...s }
})

function diagClass(text) {
  const dangerKw = ['short', 'burn', 'damage', '短路', '烧毁', '损坏']
  const warningKw = ['missing', 'unknown', '缺失', '未确定', '浮空', '开路']
  if (dangerKw.some(k => text.includes(k))) return 'diag-danger'
  if (warningKw.some(k => text.includes(k))) return 'diag-warning'
  return 'diag-info'
}

onMounted(() => {
  store.fetchAll()
})
</script>

<style scoped>
.back-link {
  color: var(--text-muted);
  text-decoration: none;
  font-size: 13px;
  display: inline-block;
  margin-bottom: 8px;
  transition: color var(--transition);
}
.back-link:hover {
  color: var(--accent);
}

.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  align-items: start;
}

.video-section {
  position: sticky;
  top: 72px;
}
.video-header {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 10px;
}
.video-container {
  width: 100%;
  aspect-ratio: 4/3;
  background: #000;
  border-radius: var(--radius);
  overflow: hidden;
}
.video-frame {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.video-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 14px;
}
.video-meta {
  display: flex;
  gap: 16px;
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-muted);
}

.metrics-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.metric-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
.metric-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.metric-label {
  font-size: 12px;
  color: var(--text-muted);
}
.metric-value-lg {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

.card-section-title {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 10px;
  color: var(--text-secondary);
}
.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.tag {
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 4px;
  font-weight: 500;
}
.tag-warning {
  background: rgba(241, 250, 140, 0.12);
  color: var(--warning);
}

.diag-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.diag-list li {
  font-size: 13px;
  padding: 6px 10px;
  border-radius: 4px;
  border-left: 3px solid var(--border);
}
.diag-danger {
  border-left-color: var(--danger) !important;
  background: rgba(255, 85, 85, 0.06);
  color: var(--danger);
}
.diag-warning {
  border-left-color: var(--warning) !important;
  background: rgba(241, 250, 140, 0.06);
  color: var(--warning);
}
.diag-info {
  color: var(--text-secondary);
}

.snapshot {
  background: var(--bg-primary);
  padding: 12px;
  border-radius: var(--radius);
  font-size: 12px;
  font-family: 'Consolas', 'Courier New', monospace;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  color: var(--text-secondary);
  max-height: 200px;
  overflow-y: auto;
}

.guidance-section {
  margin-top: 24px;
}

@media (max-width: 900px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }
  .video-section {
    position: static;
  }
}
</style>
