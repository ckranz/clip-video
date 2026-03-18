const { createApp, ref, onMounted } = Vue;

// Review View (stub)
const ReviewView = {
  props: ['brand'],
  template: `
    <div class="text-gray-400 text-center py-20">
      <p class="text-lg">Review View</p>
      <p class="text-sm mt-2">Coming in Task 9</p>
    </div>
  `
};

// Schedule View (stub)
const ScheduleView = {
  props: ['brand'],
  template: `
    <div class="text-gray-400 text-center py-20">
      <p class="text-lg">Schedule View</p>
      <p class="text-sm mt-2">Coming in Task 10</p>
    </div>
  `
};

// Tasks View (stub)
const TasksView = {
  emits: ['task-count'],
  template: `
    <div class="text-gray-400 text-center py-20">
      <p class="text-lg">Tasks View</p>
      <p class="text-sm mt-2">Coming in Task 11</p>
    </div>
  `
};

const app = createApp({
  components: {
    'review-view': ReviewView,
    'schedule-view': ScheduleView,
    'tasks-view': TasksView,
  },
  setup() {
    const currentTab = ref('review');
    const activeTaskCount = ref(0);
    const brand = ref({ name: '', logo_url: null, social_platforms: [] });
    const tabs = [
      { id: 'review', label: 'Review' },
      { id: 'schedule', label: 'Schedule' },
      { id: 'tasks', label: 'Tasks' },
    ];

    onMounted(async () => {
      try {
        const resp = await fetch('/api/brand');
        if (resp.ok) {
          brand.value = await resp.json();
          document.title = `${brand.value.name} — clip-video dashboard`;
        }
      } catch (e) {
        console.error('Failed to load brand info:', e);
      }
    });

    return { currentTab, activeTaskCount, brand, tabs };
  },
});

app.mount('#app');
