<template>
  <div class="app-layout">
    <NavBar />
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import NavBar from './components/NavBar.vue'
import { useClassroomStore } from './stores/classroom.js'
import { onMounted, onUnmounted } from 'vue'

const store = useClassroomStore()
let pollTimer = null

onMounted(() => {
  store.fetchAll()
  pollTimer = setInterval(() => store.fetchAll(), 2000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>
