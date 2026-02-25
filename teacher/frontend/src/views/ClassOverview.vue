<template>
  <div class="class-overview">
    <div class="page-header">
      <div>
        <h1>Classroom Overview</h1>
        <p>
          {{ store.onlineCount }} stations online
          <span v-if="store.lastUpdate">
            &middot; updated {{ timeAgo }}
          </span>
        </p>
      </div>
      <div class="header-actions">
        <button class="btn btn-ghost btn-sm" @click="showBroadcast = !showBroadcast">
          Broadcast
        </button>
      </div>
    </div>

    <transition name="fade">
      <div v-if="showBroadcast" class="broadcast-wrap">
        <GuidanceBox :is-broadcast="true" />
      </div>
    </transition>

    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-value">{{ store.onlineCount }}</div>
        <div class="stat-label">Online</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ store.completedCount }}</div>
        <div class="stat-label">Completed</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" :style="{ color: store.dangerStations.length > 0 ? 'var(--danger)' : '' }">
          {{ store.dangerStations.length }}
        </div>
        <div class="stat-label">Alerts</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ avgProgress }}%</div>
        <div class="stat-label">Avg Progress</div>
      </div>
    </div>

    <div v-if="store.stationList.length === 0" class="empty-state">
      <div class="icon">&#128225;</div>
      <p>No stations connected yet.</p>
      <p class="hint">Students should start their application with <code>--classroom</code> flag.</p>
    </div>

    <div v-else class="station-grid">
      <StationCard
        v-for="station in store.stationList"
        :key="station.station_id"
        :station="station"
      />
    </div>

    <div class="alert-section">
      <AlertPanel />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useClassroomStore } from '../stores/classroom.js'
import StationCard from '../components/StationCard.vue'
import AlertPanel from '../components/AlertPanel.vue'
import GuidanceBox from '../components/GuidanceBox.vue'

const store = useClassroomStore()
const showBroadcast = ref(false)

const avgProgress = computed(() => {
  const list = store.stationList.filter(s => s.online)
  if (list.length === 0) return 0
  const sum = list.reduce((a, s) => a + (s.progress || 0), 0)
  return Math.round((sum / list.length) * 100)
})

const timeAgo = computed(() => {
  if (!store.lastUpdate) return ''
  const sec = Math.round((Date.now() - store.lastUpdate.getTime()) / 1000)
  if (sec < 5) return 'just now'
  if (sec < 60) return sec + 's ago'
  return Math.floor(sec / 60) + 'm ago'
})
</script>

<style scoped>
.broadcast-wrap {
  margin-bottom: 20px;
}
.header-actions {
  display: flex;
  gap: 8px;
}
.alert-section {
  margin-top: 24px;
}
.hint {
  font-size: 13px;
  margin-top: 4px;
}
.hint code {
  background: var(--bg-card);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}
</style>
