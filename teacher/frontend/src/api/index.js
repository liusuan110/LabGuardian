import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 5000,
})

export default {
  getClassroom: () => api.get('/classroom'),
  getRanking: () => api.get('/classroom/ranking'),
  getAlerts: () => api.get('/classroom/alerts'),
  getStats: () => api.get('/classroom/stats'),
  getStation: (id) => api.get(`/station/${id}`),
  getThumbnail: (id) => api.get(`/station/${id}/thumbnail`),
  sendGuidance: (stationId, type, message, sender = 'Teacher') =>
    api.post(`/station/${stationId}/guidance`, {
      station_id: stationId,
      type,
      message,
      sender,
    }),
  broadcast: (message, sender = 'Teacher') =>
    api.post('/classroom/broadcast', { message, sender }),
  setReference: (data) => api.post('/classroom/reference', data),
  resetSession: () => api.post('/classroom/reset'),
}
