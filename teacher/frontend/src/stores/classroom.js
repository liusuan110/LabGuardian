import { defineStore } from 'pinia'
import api from '../api/index.js'

export const useClassroomStore = defineStore('classroom', {
  state: () => ({
    stations: {},
    alerts: [],
    stats: {},
    ranking: [],
    loading: false,
    error: null,
    lastUpdate: null,
  }),

  getters: {
    onlineCount: (state) =>
      Object.values(state.stations).filter(s => s.online).length,

    totalCount: (state) =>
      Object.keys(state.stations).length,

    dangerStations: (state) =>
      Object.entries(state.stations)
        .filter(([, s]) => s.risk_level === 'danger' && s.online)
        .map(([id, s]) => ({ station_id: id, ...s })),

    completedCount: (state) =>
      Object.values(state.stations).filter(s => (s.progress || 0) >= 1.0).length,

    stationList: (state) =>
      Object.entries(state.stations)
        .map(([id, s]) => ({ station_id: id, ...s }))
        .sort((a, b) => a.station_id.localeCompare(b.station_id)),
  },

  actions: {
    async fetchAll() {
      try {
        const [classRes, alertRes, statsRes, rankRes] = await Promise.all([
          api.getClassroom(),
          api.getAlerts(),
          api.getStats(),
          api.getRanking(),
        ])
        this.stations = classRes.data
        this.alerts = alertRes.data
        this.stats = statsRes.data
        this.ranking = rankRes.data
        this.lastUpdate = new Date()
        this.error = null
      } catch (err) {
        this.error = err.message
      }
    },

    async fetchRanking() {
      try {
        const res = await api.getRanking()
        this.ranking = res.data
      } catch (err) {
        this.error = err.message
      }
    },

    async sendGuidance(stationId, type, message) {
      try {
        await api.sendGuidance(stationId, type, message)
        return true
      } catch (err) {
        this.error = err.message
        return false
      }
    },

    async broadcast(message) {
      try {
        await api.broadcast(message)
        return true
      } catch (err) {
        this.error = err.message
        return false
      }
    },
  },
})
