const { createApp, ref, reactive, computed, onMounted, watch, nextTick } = Vue;

// Toast notification component
const ToastContainer = {
  setup() {
    const toasts = ref([]);
    let toastId = 0;

    function addToast(message, type = 'success') {
      const id = ++toastId;
      toasts.value.push({ id, message, type });
      setTimeout(() => {
        toasts.value = toasts.value.filter(t => t.id !== id);
      }, 2000);
    }

    // Expose globally so other components can use it
    window.__addToast = addToast;

    return { toasts };
  },
  template: `
    <div class="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      <transition-group name="toast">
        <div v-for="toast in toasts" :key="toast.id"
          :class="['px-4 py-2 rounded-lg text-sm font-medium shadow-lg',
            toast.type === 'success' ? 'bg-green-700 text-green-100' : 'bg-red-700 text-red-100']">
          {{ toast.message }}
        </div>
      </transition-group>
    </div>
  `
};

function showToast(message, type = 'success') {
  if (window.__addToast) {
    window.__addToast(message, type);
  }
}

// Utility: convert API path to media URL
// API returns e.g. "highlights/talk-a/clips/final/clip_01_final.mp4"
// Media mount is at /media pointing to the highlights dir,
// so we strip the "highlights/" prefix
function mediaUrl(path) {
  if (!path) return '';
  return '/media/' + path.replace(/^highlights\//, '');
}

// Utility: format seconds to m:ss
function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Status badge color mapping
const STATUS_STYLES = {
  new:       'bg-gray-600 text-gray-200',
  selected:  'bg-blue-600 text-blue-100',
  skipped:   'bg-gray-700 text-gray-400',
  scheduled: 'bg-amber-600 text-amber-100',
  posted:    'bg-green-600 text-green-100',
};

// Review View — main dashboard for reviewing highlight clips
const ReviewView = {
  props: ['brand'],
  setup() {
    const videos = ref([]);
    const loading = ref(true);
    const statusFilter = ref('all');
    const reprocessingVideos = reactive({});

    async function fetchVideos() {
      try {
        const resp = await fetch('/api/videos');
        if (resp.ok) {
          videos.value = await resp.json();
        }
      } catch (e) {
        console.error('Failed to load videos:', e);
      } finally {
        loading.value = false;
      }
    }

    onMounted(fetchVideos);

    const allClips = computed(() => {
      return videos.value.flatMap(v => v.clips);
    });

    const statusCounts = computed(() => {
      const counts = { all: 0, new: 0, selected: 0, skipped: 0, scheduled: 0, posted: 0 };
      for (const clip of allClips.value) {
        counts.all++;
        if (counts[clip.status] !== undefined) {
          counts[clip.status]++;
        }
      }
      return counts;
    });

    const filteredVideos = computed(() => {
      if (statusFilter.value === 'all') return videos.value;
      return videos.value
        .map(v => ({
          ...v,
          clips: v.clips.filter(c => c.status === statusFilter.value),
        }))
        .filter(v => v.clips.length > 0);
    });

    // Track which version each clip is showing
    const clipVersions = reactive({});

    function getClipVersion(clipId) {
      return clipVersions[clipId] || 'final';
    }

    function setClipVersion(clipId, version) {
      clipVersions[clipId] = version;
    }

    function getClipSrc(clip) {
      const version = getClipVersion(clip.clip_id);
      if (version === 'raw') return mediaUrl(clip.raw_path);
      if (version === 'portrait') return mediaUrl(clip.portrait_path);
      return mediaUrl(clip.final_path);
    }

    // Track expanded social copy sections
    const expandedCopy = reactive({});

    function toggleCopy(clipId) {
      expandedCopy[clipId] = !expandedCopy[clipId];
    }

    async function copyToClipboard(clip) {
      const text = [
        clip.hook_text,
        '',
        clip.summary,
        '',
        clip.topics.map(t => '#' + t.replace(/\s+/g, '')).join(' '),
      ].join('\n');
      try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard');
      } catch (e) {
        showToast('Failed to copy', 'error');
      }
    }

    // Video metadata updates
    async function updateVideo(filename, field, value) {
      try {
        const body = {};
        body[field] = value;
        const resp = await fetch(`/api/videos/${encodeURIComponent(filename)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (resp.ok) {
          showToast('Saved');
        } else {
          showToast('Save failed', 'error');
        }
      } catch (e) {
        showToast('Save failed', 'error');
      }
    }

    function onSpeakerBlur(video, event) {
      const val = event.target.value.trim();
      if (val !== video.speaker) {
        video.speaker = val;
        updateVideo(video.filename, 'speaker', val);
      }
    }

    function onPositionChange(video, event) {
      const val = event.target.value;
      const oldVal = video.speaker_position;
      video.speaker_position = val;
      updateVideo(video.filename, 'speaker_position', val);
      if (oldVal !== val) {
        reprocessingVideos[video.filename] = true;
      }
    }

    function onYoutubeBlur(video, event) {
      const val = event.target.value.trim();
      if (val !== video.youtube_url) {
        video.youtube_url = val;
        updateVideo(video.filename, 'youtube_url', val);
      }
    }

    async function reprocessVideo(video) {
      try {
        const resp = await fetch(`/api/videos/${encodeURIComponent(video.filename)}/reprocess`, {
          method: 'POST',
        });
        if (resp.ok) {
          reprocessingVideos[video.filename] = false;
          showToast('Reprocessing started');
        } else {
          showToast('Reprocess failed', 'error');
        }
      } catch (e) {
        showToast('Reprocess failed', 'error');
      }
    }

    // Clip status updates
    async function setClipStatus(clip, status) {
      const oldStatus = clip.status;
      clip.status = status;
      try {
        const resp = await fetch(`/api/clips/${encodeURIComponent(clip.clip_id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status }),
        });
        if (resp.ok) {
          showToast(status === 'selected' ? 'Selected' : 'Skipped');
        } else {
          clip.status = oldStatus;
          showToast('Update failed', 'error');
        }
      } catch (e) {
        clip.status = oldStatus;
        showToast('Update failed', 'error');
      }
    }

    const statusFilters = ['all', 'new', 'selected', 'skipped', 'scheduled', 'posted'];

    function statusClass(status) {
      return STATUS_STYLES[status] || STATUS_STYLES.new;
    }

    return {
      videos, loading, statusFilter, statusFilters, statusCounts,
      filteredVideos, reprocessingVideos,
      getClipVersion, setClipVersion, getClipSrc,
      expandedCopy, toggleCopy, copyToClipboard,
      onSpeakerBlur, onPositionChange, onYoutubeBlur, reprocessVideo,
      setClipStatus, formatDuration, statusClass,
    };
  },
  template: `
    <div>
      <!-- Loading state -->
      <div v-if="loading" class="text-gray-400 text-center py-20">
        <p class="text-lg">Loading clips...</p>
      </div>

      <div v-else>
        <!-- Filter bar -->
        <div class="flex flex-wrap gap-2 mb-6">
          <button v-for="sf in statusFilters" :key="sf"
            @click="statusFilter = sf"
            :class="['px-3 py-1.5 rounded-lg text-sm font-medium transition-colors capitalize',
              statusFilter === sf
                ? 'bg-gray-700 text-white'
                : 'bg-gray-900 text-gray-400 hover:text-gray-200 hover:bg-gray-800']">
            {{ sf }}
            <span class="ml-1 text-xs opacity-70">({{ statusCounts[sf] }})</span>
          </button>
        </div>

        <!-- Empty state -->
        <div v-if="filteredVideos.length === 0" class="text-gray-500 text-center py-16">
          <p class="text-lg">No clips match this filter.</p>
        </div>

        <!-- Videos grouped by speaker -->
        <div v-for="video in filteredVideos" :key="video.filename" class="mb-10">
          <!-- Video header -->
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-4">
            <div class="flex flex-wrap items-center gap-3">
              <!-- Speaker name -->
              <input type="text" :value="video.speaker" @blur="onSpeakerBlur(video, $event)"
                placeholder="Speaker name"
                class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-100 w-48 focus:outline-none focus:border-blue-500">

              <!-- Speaker position -->
              <select :value="video.speaker_position || 'left'"
                @change="onPositionChange(video, $event)"
                class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-blue-500">
                <option value="left">Left</option>
                <option value="center">Center</option>
                <option value="right">Right</option>
              </select>

              <!-- YouTube URL -->
              <input type="text" :value="video.youtube_url || ''" @blur="onYoutubeBlur(video, $event)"
                placeholder="YouTube URL"
                class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-100 flex-1 min-w-[200px] focus:outline-none focus:border-blue-500">

              <!-- YouTube link icon -->
              <a v-if="video.youtube_url" :href="video.youtube_url" target="_blank" rel="noopener"
                class="text-gray-400 hover:text-red-400 transition-colors" title="Open on YouTube">
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                </svg>
              </a>

              <!-- Reprocess button -->
              <button v-if="reprocessingVideos[video.filename]"
                @click="reprocessVideo(video)"
                class="px-3 py-1.5 rounded-lg text-sm font-medium bg-amber-700 hover:bg-amber-600 text-amber-100 transition-colors">
                Reprocess
              </button>

              <!-- Filename label -->
              <span class="text-xs text-gray-500 ml-auto">{{ video.filename }}</span>
            </div>
          </div>

          <!-- Clip cards grid -->
          <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
            <div v-for="clip in video.clips" :key="clip.clip_id"
              class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">

              <!-- Video player -->
              <div class="relative">
                <video controls preload="metadata" :key="getClipSrc(clip)"
                  class="w-full aspect-video bg-black object-contain">
                  <source :src="getClipSrc(clip)" type="video/mp4">
                </video>

                <!-- Version toggle overlay -->
                <div class="absolute top-2 right-2 flex gap-1">
                  <button v-for="ver in ['raw', 'portrait', 'final']" :key="ver"
                    @click="setClipVersion(clip.clip_id, ver)"
                    :class="['px-2 py-0.5 rounded text-xs font-medium capitalize transition-colors',
                      getClipVersion(clip.clip_id) === ver
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-800/80 text-gray-300 hover:bg-gray-700/80']">
                    {{ ver }}
                  </button>
                </div>
              </div>

              <!-- Card body -->
              <div class="p-4 space-y-3">
                <!-- Status badge + quality + duration row -->
                <div class="flex items-center gap-2 flex-wrap">
                  <span :class="['px-2 py-0.5 rounded-full text-xs font-medium capitalize',
                    statusClass(clip.status)]"
                    v-text="clip.status"></span>
                  <span class="text-xs text-gray-500">{{ formatDuration(clip.duration) }}</span>
                  <span class="text-xs text-gray-500 ml-auto" :title="'Quality: ' + clip.quality_score">
                    {{ Math.round(clip.quality_score * 100) }}%
                  </span>
                </div>

                <!-- Hook text -->
                <p class="text-sm font-medium text-gray-200 leading-snug">{{ clip.hook_text }}</p>

                <!-- Summary -->
                <p class="text-xs text-gray-400 leading-relaxed">{{ clip.summary }}</p>

                <!-- Topics -->
                <div class="flex flex-wrap gap-1">
                  <span v-for="topic in clip.topics" :key="topic"
                    class="px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 text-xs">
                    {{ topic }}
                  </span>
                </div>

                <!-- Social copy section -->
                <div>
                  <button @click="toggleCopy(clip.clip_id)"
                    class="text-xs text-gray-500 hover:text-gray-300 transition-colors">
                    {{ expandedCopy[clip.clip_id] ? 'Hide social copy' : 'Show social copy' }}
                  </button>
                  <div v-if="expandedCopy[clip.clip_id]" class="mt-2 p-3 bg-gray-800 rounded-lg">
                    <p class="text-sm font-medium text-gray-200 mb-1">{{ clip.hook_text }}</p>
                    <p class="text-xs text-gray-400 mb-2">{{ clip.summary }}</p>
                    <p class="text-xs text-blue-400 mb-3">{{ clip.topics.map(t => '#' + t.replace(/\\s+/g, '')).join(' ') }}</p>
                    <button @click="copyToClipboard(clip)"
                      class="px-3 py-1 rounded-lg text-xs font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors">
                      Copy
                    </button>
                  </div>
                </div>

                <!-- Action buttons -->
                <div class="flex gap-2 pt-1">
                  <button @click="setClipStatus(clip, 'selected')"
                    :class="['flex-1 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                      clip.status === 'selected'
                        ? 'bg-blue-700 text-blue-100'
                        : 'bg-blue-600 hover:bg-blue-500 text-white']">
                    Select
                  </button>
                  <button @click="setClipStatus(clip, 'skipped')"
                    :class="['flex-1 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                      clip.status === 'skipped'
                        ? 'bg-gray-600 text-gray-300'
                        : 'bg-gray-700 hover:bg-gray-600 text-gray-300']">
                    Skip
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
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
    'toast-container': ToastContainer,
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
