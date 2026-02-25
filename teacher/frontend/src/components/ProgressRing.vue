<template>
  <div class="progress-ring" :style="{ width: size + 'px', height: size + 'px' }">
    <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`">
      <circle
        class="ring-bg"
        :cx="center"
        :cy="center"
        :r="radius"
        fill="none"
        :stroke-width="strokeWidth"
      />
      <circle
        class="ring-fill"
        :cx="center"
        :cy="center"
        :r="radius"
        fill="none"
        :stroke-width="strokeWidth"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset"
        :stroke="strokeColor"
        stroke-linecap="round"
      />
    </svg>
    <div class="ring-label">
      <span class="ring-value">{{ displayValue }}</span>
      <span v-if="showUnit" class="ring-unit">%</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  value: { type: Number, default: 0 },
  size: { type: Number, default: 56 },
  strokeWidth: { type: Number, default: 4 },
  showUnit: { type: Boolean, default: true },
})

const center = computed(() => props.size / 2)
const radius = computed(() => (props.size - props.strokeWidth) / 2 - 1)
const circumference = computed(() => 2 * Math.PI * radius.value)
const clampedValue = computed(() => Math.min(1, Math.max(0, props.value)))
const dashOffset = computed(() => circumference.value * (1 - clampedValue.value))
const displayValue = computed(() => Math.round(clampedValue.value * 100))

const strokeColor = computed(() => {
  if (clampedValue.value >= 1.0) return 'var(--success)'
  if (clampedValue.value >= 0.6) return 'var(--accent)'
  if (clampedValue.value >= 0.3) return 'var(--orange)'
  return 'var(--danger)'
})
</script>

<style scoped>
.progress-ring {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.progress-ring svg {
  transform: rotate(-90deg);
  position: absolute;
}
.ring-bg {
  stroke: var(--border);
}
.ring-fill {
  transition: stroke-dashoffset 0.6s ease, stroke 0.3s ease;
}
.ring-label {
  display: flex;
  align-items: baseline;
  gap: 1px;
  z-index: 1;
}
.ring-value {
  font-size: 14px;
  font-weight: 700;
  color: var(--text-primary);
}
.ring-unit {
  font-size: 10px;
  color: var(--text-muted);
}
</style>
