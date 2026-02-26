<template>
  <div class="guidance-box card">
    <div class="box-header">
      <span class="box-title">Send Guidance</span>
    </div>

    <div class="quick-hints">
      <button
        v-for="hint in quickHints"
        :key="hint.text"
        class="btn btn-ghost btn-sm"
        @click="sendQuick(hint)"
      >
        {{ hint.label }}
      </button>
    </div>

    <div class="input-row">
      <select v-model="msgType" class="type-select">
        <option value="hint">Hint</option>
        <option value="warning">Warning</option>
        <option value="approve">Approve</option>
      </select>
      <input
        v-model="message"
        type="text"
        :placeholder="placeholder"
        @keyup.enter="send"
      />
      <button class="btn btn-primary" @click="send" :disabled="!message.trim()">
        Send
      </button>
    </div>

    <div v-if="status" class="send-status" :class="status">
      {{ statusText }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useClassroomStore } from '../stores/classroom.js'

const props = defineProps({
  stationId: { type: String, default: '' },
  isBroadcast: { type: Boolean, default: false },
})

const store = useClassroomStore()
const message = ref('')
const msgType = ref('hint')
const status = ref('')
const statusText = ref('')

const placeholder = computed(() =>
  props.isBroadcast ? 'Broadcast to all students...' : 'Type guidance message...'
)

const quickHints = [
  { label: 'Check polarity', text: 'Please check your component polarity carefully.', type: 'hint' },
  { label: 'Missing component', text: 'You may be missing a component. Check the reference circuit.', type: 'hint' },
  { label: 'Short circuit risk', text: 'Warning: potential short circuit detected. Please review your wiring immediately.', type: 'warning' },
  { label: 'Good job', text: 'Your circuit looks correct. Great work!', type: 'approve' },
]

async function sendQuick(hint) {
  message.value = hint.text
  msgType.value = hint.type
  await send()
}

async function send() {
  const text = message.value.trim()
  if (!text) return

  let ok
  if (props.isBroadcast) {
    ok = await store.broadcast(text)
  } else {
    ok = await store.sendGuidance(props.stationId, msgType.value, text)
  }

  if (ok) {
    status.value = 'success'
    statusText.value = 'Sent!'
    message.value = ''
  } else {
    status.value = 'error'
    statusText.value = 'Failed to send'
  }

  setTimeout(() => { status.value = '' }, 2000)
}
</script>

<style scoped>
.box-header {
  margin-bottom: 10px;
}
.box-title {
  font-weight: 600;
  font-size: 14px;
}
.quick-hints {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}
.input-row {
  display: flex;
  gap: 8px;
}
.type-select {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-primary);
  padding: 8px 10px;
  font-size: 13px;
  outline: none;
  width: 100px;
  flex-shrink: 0;
}
.input-row input {
  flex: 1;
  min-width: 0;
}
.send-status {
  margin-top: 8px;
  font-size: 12px;
  font-weight: 500;
}
.send-status.success {
  color: var(--success);
}
.send-status.error {
  color: var(--danger);
}
</style>
