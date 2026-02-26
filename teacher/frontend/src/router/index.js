import { createRouter, createWebHashHistory } from 'vue-router'
import ClassOverview from '../views/ClassOverview.vue'
import StationDetail from '../views/StationDetail.vue'
import Ranking from '../views/Ranking.vue'
import Statistics from '../views/Statistics.vue'

const routes = [
  { path: '/', name: 'overview', component: ClassOverview },
  { path: '/station/:id', name: 'station', component: StationDetail, props: true },
  { path: '/ranking', name: 'ranking', component: Ranking },
  { path: '/stats', name: 'stats', component: Statistics },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
