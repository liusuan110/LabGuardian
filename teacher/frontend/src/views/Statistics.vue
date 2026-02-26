<template>
  <div class="statistics-page">
    <div class="page-header">
      <div>
        <h1>Class Statistics</h1>
        <p>Aggregated metrics for the current session</p>
      </div>
      <div class="header-actions">
        <button class="btn btn-ghost btn-sm" @click="store.fetchAll()">Refresh</button>
      </div>
    </div>

    <!-- Overview Cards -->
    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-value">{{ stats.total_stations || 0 }}</div>
        <div class="stat-label">Total Stations</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color: var(--success)">{{ stats.online_count || 0 }}</div>
        <div class="stat-label">Online</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ avgProgress }}%</div>
        <div class="stat-label">Avg Progress</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color: var(--success)">{{ stats.completed_count || 0 }}</div>
        <div class="stat-label">Completed</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color: var(--danger)">{{ stats.total_risk_events || 0 }}</div>
        <div class="stat-label">Risk Events</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color: var(--warning)">{{ stats.danger_count || 0 }}</div>
        <div class="stat-label">Active Alerts</div>
      </div>
    </div>

    <!-- Progress Distribution -->
    <div class="section">
      <div class="section-header">
        <span class="section-title">Progress Distribution</span>
      </div>
      <div class="card">
        <div class="bar-chart">
          <div
            v-for="bucket in progressBuckets"
            :key="bucket.label"
            class="bar-item"
          >
            <div class="bar-wrapper">
              <div
                class="bar"
                :style="{ height: bucket.height + '%' }"
                :class="bucket.class"
              ></div>
            </div>
            <div class="bar-label">{{ bucket.label }}</div>
            <div class="bar-count">{{ bucket.count }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Error Distribution -->
    <div class="section" v-if="errorEntries.length > 0">
      <div class="section-header">
        <span class="section-title">Top Error Types</span>
      </div>
      <div class="card">
        <div class="error-bars">
          <div
            v-for="entry in errorEntries"
            :key="entry.type"
            class="error-bar-row"
          >
            <span class="error-type">{{ entry.type }}</span>
            <div class="error-bar-track">
              <div
                class="error-bar-fill"
                :style="{ width: entry.pct + '%' }"
              ></div>
            </div>
            <span class="error-count">{{ entry.count }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Station Status Table -->
    <div class="section">
      <div class="section-header">
        <span class="section-title">Station Status Summary</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Station</th>
              <th>Student</th>
              <th>Progress</th>
              <th>Risk Level</th>
              <th>Risk Events</th>
              <th>Online</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in stationList" :key="s.station_id">
              <td>
                <router-link
                  :to="{ name: 'station', params: { id: s.station_id } }"
                  class="station-link"
                >
                  {{ s.station_id }}
                </router-link>
              </td>
              <td class="text-secondary">{{ s.student_name || '--' }}</td>
              <td>
                <div class="inline-progress">
                  <div class="progress-bar" style="width: 80px">
                    <div
                      class="fill"
                      :class="{ complete: (s.progress || 0) >= 1.0 }"
                      :style="{ width: Math.round((s.progress || 0) * 100) + '%' }"
                    ></div>
                  </div>
                  <span>{{ Math.round((s.progress || 0) * 100) }}%</span>
                </div>
              </td>
              <td>
                <span class="badge" :class="'badge-' + (s.risk_level || 'safe')">
                  {{ s.risk_level || 'safe' }}
                </span>
              </td>
              <td>{{ s.risk_event_count || 0 }}</td>
              <td>
                <span class="status-dot" :class="s.online ? 'on' : 'off'"></span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useClassroomStore } from '../stores/classroom.js'

const store = useClassroomStore()

const stats = computed(() => store.stats || {})
const stationList = computed(() => store.stationList || [])

const avgProgress = computed(() => {
  return Math.round((stats.value.avg_progress || 0) * 100)
})

const progressBuckets = computed(() => {
  const list = stationList.value
  const buckets = [
    { label: '0-20%', min: 0, max: 0.2, count: 0, class: 'bar-danger' },
    { label: '20-40%', min: 0.2, max: 0.4, count: 0, class: 'bar-warning' },
    { label: '40-60%', min: 0.4, max: 0.6, count: 0, class: 'bar-orange' },
    { label: '60-80%', min: 0.6, max: 0.8, count: 0, class: 'bar-accent' },
    { label: '80-100%', min: 0.8, max: 1.01, count: 0, class: 'bar-success' },
  ]
  for (const s of list) {
    const p = s.progress || 0
    for (const b of buckets) {
      if (p >= b.min && p < b.max) { b.count++; break }
    }
  }
  const maxCount = Math.max(1, ...buckets.map(b => b.count))
  return buckets.map(b => ({
    ...b,
    height: Math.round((b.count / maxCount) * 100),
  }))
})

const errorEntries = computed(() => {
  const hist = stats.value.error_histogram || {}
  const entries = Object.entries(hist)
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8)
  const maxCount = Math.max(1, ...entries.map(e => e.count))
  return entries.map(e => ({
    ...e,
    pct: Math.round((e.count / maxCount) * 100),
  }))
})

onMounted(() => {
  store.fetchAll()
})
</script>

<style scoped>
.section {
  margin-bottom: 28px;
}
.header-actions {
  display: flex;
  gap: 8px;
}

/* Bar chart */
.bar-chart {
  display: flex;
  align-items: flex-end;
  justify-content: space-around;
  height: 160px;
  padding: 0 20px;
  gap: 12px;
}
.bar-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  max-width: 80px;
}
.bar-wrapper {
  width: 100%;
  height: 120px;
  display: flex;
  align-items: flex-end;
}
.bar {
  width: 100%;
  border-radius: 4px 4px 0 0;
  min-height: 4px;
  transition: height 0.5s ease;
}
.bar-danger { background: var(--danger); }
.bar-warning { background: var(--warning); }
.bar-orange { background: var(--orange); }
.bar-accent { background: var(--accent); }
.bar-success { background: var(--success); }
.bar-label {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 6px;
}
.bar-count {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
}

/* Error bars */
.error-bars {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.error-bar-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.error-type {
  width: 180px;
  font-size: 13px;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex-shrink: 0;
}
.error-bar-track {
  flex: 1;
  height: 8px;
  background: var(--bg-primary);
  border-radius: 4px;
  overflow: hidden;
}
.error-bar-fill {
  height: 100%;
  background: var(--danger);
  border-radius: 4px;
  transition: width 0.5s ease;
}
.error-count {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  min-width: 30px;
  text-align: right;
}

/* Table */
.station-link {
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
}
.station-link:hover {
  text-decoration: underline;
}
.text-secondary {
  color: var(--text-secondary);
}
.inline-progress {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
}
.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.status-dot.on {
  background: var(--success);
  box-shadow: 0 0 6px rgba(80, 250, 123, 0.4);
}
.status-dot.off {
  background: var(--text-muted);
}
</style>
