<template>
  <div class="ranking-page">
    <div class="page-header">
      <div>
        <h1>Progress Ranking</h1>
        <p>Sorted by circuit completion progress</p>
      </div>
      <div class="header-actions">
        <button class="btn btn-ghost btn-sm" @click="fetchRanking">Refresh</button>
        <button class="btn btn-ghost btn-sm" @click="exportCSV">Export CSV</button>
      </div>
    </div>

    <div v-if="ranking.length === 0" class="empty-state">
      <div class="icon">&#127942;</div>
      <p>No ranking data available yet.</p>
    </div>

    <div v-else class="table-wrap">
      <table>
        <thead>
          <tr>
            <th style="width: 50px">#</th>
            <th>Station</th>
            <th>Student</th>
            <th style="width: 200px">Progress</th>
            <th>Similarity</th>
            <th>Match</th>
            <th>Components</th>
            <th>Risk Events</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(item, index) in ranking"
            :key="item.station_id"
            :class="{ 'row-danger': item.risk_level === 'danger' }"
          >
            <td class="rank-cell">
              <span class="rank-num" :class="rankClass(index)">{{ index + 1 }}</span>
            </td>
            <td>
              <router-link
                :to="{ name: 'station', params: { id: item.station_id } }"
                class="station-link"
              >
                {{ item.station_id }}
              </router-link>
            </td>
            <td class="text-secondary">{{ item.student_name || '--' }}</td>
            <td>
              <div class="progress-cell">
                <div class="progress-bar">
                  <div
                    class="fill"
                    :class="{ complete: (item.progress || 0) >= 1.0 }"
                    :style="{ width: Math.round((item.progress || 0) * 100) + '%' }"
                  ></div>
                </div>
                <span class="progress-text">{{ Math.round((item.progress || 0) * 100) }}%</span>
              </div>
            </td>
            <td>{{ ((item.similarity || 0) * 100).toFixed(1) }}%</td>
            <td>{{ item.match_level || '--' }}</td>
            <td>{{ item.component_count || 0 }}</td>
            <td>
              <span v-if="item.risk_event_count > 0" class="risk-count">
                {{ item.risk_event_count }}
              </span>
              <span v-else class="text-muted">0</span>
            </td>
            <td>
              <span class="badge" :class="'badge-' + (item.online ? item.risk_level : 'offline')">
                {{ item.online ? (item.risk_level || 'safe') : 'offline' }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useClassroomStore } from '../stores/classroom.js'

const store = useClassroomStore()

const ranking = computed(() => store.ranking || [])

function rankClass(index) {
  if (index === 0) return 'gold'
  if (index === 1) return 'silver'
  if (index === 2) return 'bronze'
  return ''
}

async function fetchRanking() {
  await store.fetchRanking()
}

function exportCSV() {
  const headers = ['Rank', 'Station', 'Student', 'Progress', 'Similarity', 'Match', 'Components', 'RiskEvents', 'Status']
  const rows = ranking.value.map((item, i) => [
    i + 1,
    item.station_id,
    item.student_name || '',
    Math.round((item.progress || 0) * 100) + '%',
    ((item.similarity || 0) * 100).toFixed(1) + '%',
    item.match_level || '',
    item.component_count || 0,
    item.risk_event_count || 0,
    item.online ? (item.risk_level || 'safe') : 'offline',
  ])

  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `ranking_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

onMounted(() => {
  fetchRanking()
})
</script>

<style scoped>
.header-actions {
  display: flex;
  gap: 8px;
}
.rank-cell {
  text-align: center;
}
.rank-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  font-weight: 700;
  font-size: 13px;
  background: var(--bg-primary);
}
.rank-num.gold {
  background: rgba(255, 184, 108, 0.2);
  color: var(--orange);
}
.rank-num.silver {
  background: rgba(189, 147, 249, 0.15);
  color: var(--purple);
}
.rank-num.bronze {
  background: rgba(139, 233, 253, 0.15);
  color: var(--info);
}

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
.text-muted {
  color: var(--text-muted);
}

.progress-cell {
  display: flex;
  align-items: center;
  gap: 10px;
}
.progress-cell .progress-bar {
  flex: 1;
}
.progress-text {
  font-size: 13px;
  font-weight: 600;
  min-width: 40px;
  text-align: right;
}

.risk-count {
  color: var(--danger);
  font-weight: 600;
}

.row-danger td {
  background: rgba(255, 85, 85, 0.04);
}
</style>
