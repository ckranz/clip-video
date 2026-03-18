const { createApp, ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } = Vue;

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

// Schedule View — monthly calendar for scheduling clips
const ScheduleView = {
  props: ['brand'],
  setup(props) {
    const schedule = ref([]);
    const videos = ref([]);
    const loading = ref(true);
    const currentYear = ref(new Date().getFullYear());
    const currentMonth = ref(new Date().getMonth()); // 0-indexed
    const scheduleModal = ref(false);
    const selectedDate = ref(null);
    const selectedClipId = ref(null);
    const selectedPlatform = ref('linkedin');
    const detailEntry = ref(null);

    const MONTH_NAMES = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];

    const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    async function fetchSchedule() {
      try {
        const resp = await fetch('/api/schedule');
        if (resp.ok) schedule.value = await resp.json();
      } catch (e) {
        console.error('Failed to load schedule:', e);
      }
    }

    async function fetchVideos() {
      try {
        const resp = await fetch('/api/videos');
        if (resp.ok) videos.value = await resp.json();
      } catch (e) {
        console.error('Failed to load videos:', e);
      }
    }

    onMounted(async () => {
      await Promise.all([fetchSchedule(), fetchVideos()]);
      loading.value = false;
    });

    const availableClips = computed(() => {
      return videos.value
        .flatMap(v => v.clips.map(c => ({ ...c, speaker: v.speaker })))
        .filter(c => c.status === 'selected');
    });

    const monthLabel = computed(() => {
      return `${MONTH_NAMES[currentMonth.value]} ${currentYear.value}`;
    });

    const calendarDays = computed(() => {
      const year = currentYear.value;
      const month = currentMonth.value;
      const firstDay = new Date(year, month, 1);
      const lastDay = new Date(year, month + 1, 0);

      // Monday=0 .. Sunday=6 (ISO week)
      let startDow = firstDay.getDay() - 1;
      if (startDow < 0) startDow = 6;

      const days = [];

      // Days from previous month to fill the first week
      const prevMonthLast = new Date(year, month, 0).getDate();
      for (let i = startDow - 1; i >= 0; i--) {
        const d = prevMonthLast - i;
        const m = month === 0 ? 11 : month - 1;
        const y = month === 0 ? year - 1 : year;
        days.push({ date: d, month: m, year: y, outside: true });
      }

      // Days in current month
      for (let d = 1; d <= lastDay.getDate(); d++) {
        days.push({ date: d, month, year, outside: false });
      }

      // Fill remaining cells to complete the grid (up to 42 = 6 rows)
      const remaining = (7 - (days.length % 7)) % 7;
      const nextMonth = month === 11 ? 0 : month + 1;
      const nextYear = month === 11 ? year + 1 : year;
      for (let d = 1; d <= remaining; d++) {
        days.push({ date: d, month: nextMonth, year: nextYear, outside: true });
      }

      return days;
    });

    function dateKey(year, month, date) {
      const m = String(month + 1).padStart(2, '0');
      const d = String(date).padStart(2, '0');
      return `${year}-${m}-${d}`;
    }

    function entriesForDay(day) {
      const key = dateKey(day.year, day.month, day.date);
      return schedule.value.filter(e => e.date === key);
    }

    function isToday(day) {
      const now = new Date();
      return day.date === now.getDate() && day.month === now.getMonth() && day.year === now.getFullYear();
    }

    function prevMonth() {
      if (currentMonth.value === 0) {
        currentMonth.value = 11;
        currentYear.value--;
      } else {
        currentMonth.value--;
      }
    }

    function nextMonth() {
      if (currentMonth.value === 11) {
        currentMonth.value = 0;
        currentYear.value++;
      } else {
        currentMonth.value++;
      }
    }

    function platformColor(platform) {
      if (platform === 'linkedin') return 'bg-blue-600 text-blue-100';
      if (platform === 'youtube') return 'bg-red-600 text-red-100';
      return 'bg-gray-600 text-gray-200';
    }

    function openScheduleModal(day) {
      if (day.outside) return;
      selectedDate.value = dateKey(day.year, day.month, day.date);
      selectedClipId.value = availableClips.value.length > 0 ? availableClips.value[0].clip_id : null;
      selectedPlatform.value = (props.brand && props.brand.social_platforms && props.brand.social_platforms[0]) || 'linkedin';
      scheduleModal.value = true;
    }

    async function submitSchedule() {
      if (!selectedClipId.value || !selectedDate.value) return;
      try {
        const resp = await fetch(`/api/clips/${encodeURIComponent(selectedClipId.value)}/schedule`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ platform: selectedPlatform.value, date: selectedDate.value }),
        });
        if (resp.ok) {
          scheduleModal.value = false;
          await fetchSchedule();
          showToast('Clip scheduled');
        } else {
          showToast('Failed to schedule', 'error');
        }
      } catch (e) {
        showToast('Failed to schedule', 'error');
      }
    }

    async function removeSchedule(entry) {
      // Find the index of this schedule entry on the clip
      const clipEntries = schedule.value.filter(e => e.clip_id === entry.clip_id);
      const idx = clipEntries.indexOf(entry);
      if (idx < 0) return;
      // The API uses the index within the clip's schedule array, which we need
      // to compute by counting matching entries by date+platform order
      try {
        const resp = await fetch(`/api/clips/${encodeURIComponent(entry.clip_id)}/schedule/${idx}`, {
          method: 'DELETE',
        });
        if (resp.ok) {
          await fetchSchedule();
          detailEntry.value = null;
          showToast('Schedule removed');
        } else {
          showToast('Failed to remove', 'error');
        }
      } catch (e) {
        showToast('Failed to remove', 'error');
      }
    }

    function showDetail(entry) {
      detailEntry.value = entry;
    }

    function closeDetail() {
      detailEntry.value = null;
    }

    const platforms = computed(() => {
      return (props.brand && props.brand.social_platforms && props.brand.social_platforms.length > 0)
        ? props.brand.social_platforms
        : ['linkedin'];
    });

    return {
      schedule, videos, loading, currentYear, currentMonth, scheduleModal,
      selectedDate, selectedClipId, selectedPlatform, detailEntry,
      monthLabel, calendarDays, availableClips, platforms,
      DAY_NAMES, entriesForDay, isToday,
      prevMonth, nextMonth, platformColor,
      openScheduleModal, submitSchedule, removeSchedule,
      showDetail, closeDetail,
    };
  },
  template: `
    <div>
      <div v-if="loading" class="text-gray-400 text-center py-20">
        <p class="text-lg">Loading schedule...</p>
      </div>

      <div v-else>
        <!-- Month navigation header -->
        <div class="flex items-center justify-center gap-4 mb-6">
          <button @click="prevMonth"
            class="px-3 py-1.5 rounded-lg text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors">
            &lt;
          </button>
          <h2 class="text-xl font-semibold text-gray-100 min-w-[200px] text-center">{{ monthLabel }}</h2>
          <button @click="nextMonth"
            class="px-3 py-1.5 rounded-lg text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors">
            &gt;
          </button>
        </div>

        <!-- Calendar grid (hidden on small screens) -->
        <div class="hidden md:grid grid-cols-7 gap-px bg-gray-800 rounded-xl overflow-hidden border border-gray-800">
          <!-- Day-of-week headers -->
          <div v-for="day in DAY_NAMES" :key="day"
            class="bg-gray-900 px-2 py-2 text-center text-xs font-medium text-gray-400 uppercase">
            {{ day }}
          </div>

          <!-- Day cells -->
          <div v-for="(day, i) in calendarDays" :key="i"
            @click="openScheduleModal(day)"
            :class="[
              'bg-gray-900 p-2 min-h-[100px] cursor-pointer transition-colors hover:bg-gray-800/80',
              day.outside ? 'opacity-40' : '',
              isToday(day) ? 'ring-2 ring-inset ring-blue-500' : '',
              !day.outside && entriesForDay(day).length === 0 ? 'text-gray-600' : '',
            ]">
            <div class="text-xs font-medium mb-1" :class="day.outside ? 'text-gray-700' : 'text-gray-300'">
              {{ day.date }}
            </div>
            <div v-for="entry in entriesForDay(day)" :key="entry.clip_id + entry.platform + entry.date"
              @click.stop="showDetail(entry)"
              :class="['rounded px-1.5 py-0.5 text-xs mb-1 flex items-center justify-between gap-1 cursor-pointer truncate',
                platformColor(entry.platform)]">
              <span class="truncate">{{ entry.speaker || 'Clip' }}</span>
              <button @click.stop="removeSchedule(entry)" class="flex-shrink-0 opacity-60 hover:opacity-100">&times;</button>
            </div>
          </div>
        </div>

        <!-- List view for small screens -->
        <div class="md:hidden space-y-2">
          <div v-if="schedule.length === 0" class="text-gray-500 text-center py-8">
            No clips scheduled this month.
          </div>
          <div v-for="entry in schedule" :key="entry.clip_id + entry.date + entry.platform"
            @click="showDetail(entry)"
            class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex items-center justify-between gap-3 cursor-pointer">
            <div class="flex-1 min-w-0">
              <div class="text-sm font-medium text-gray-200 truncate">{{ entry.speaker || 'Clip' }}</div>
              <div class="text-xs text-gray-400">{{ entry.date }}</div>
            </div>
            <span :class="['px-2 py-0.5 rounded text-xs font-medium capitalize', platformColor(entry.platform)]">
              {{ entry.platform }}
            </span>
            <button @click.stop="removeSchedule(entry)" class="text-gray-500 hover:text-red-400">&times;</button>
          </div>
        </div>

        <!-- Detail overlay -->
        <div v-if="detailEntry" class="fixed inset-0 z-40 flex items-center justify-center bg-black/60" @click.self="closeDetail">
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-6 max-w-md w-full mx-4 space-y-3">
            <div class="flex items-center justify-between">
              <h3 class="text-lg font-semibold text-gray-100">Scheduled Clip</h3>
              <button @click="closeDetail" class="text-gray-400 hover:text-gray-200">&times;</button>
            </div>
            <div class="space-y-2">
              <p class="text-sm text-gray-300"><span class="text-gray-500">Speaker:</span> {{ detailEntry.speaker }}</p>
              <p class="text-sm text-gray-300"><span class="text-gray-500">Date:</span> {{ detailEntry.date }}</p>
              <p class="text-sm text-gray-300"><span class="text-gray-500">Platform:</span> {{ detailEntry.platform }}</p>
              <p class="text-sm font-medium text-gray-200">{{ detailEntry.hook_text }}</p>
              <p class="text-xs text-gray-400">{{ detailEntry.summary }}</p>
            </div>
            <div class="flex justify-end gap-2 pt-2">
              <button @click="removeSchedule(detailEntry)"
                class="px-3 py-1.5 rounded-lg text-sm font-medium bg-red-700 hover:bg-red-600 text-red-100 transition-colors">
                Remove
              </button>
              <button @click="closeDetail"
                class="px-3 py-1.5 rounded-lg text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors">
                Close
              </button>
            </div>
          </div>
        </div>

        <!-- Schedule modal -->
        <div v-if="scheduleModal" class="fixed inset-0 z-40 flex items-center justify-center bg-black/60" @click.self="scheduleModal = false">
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-6 max-w-md w-full mx-4 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-lg font-semibold text-gray-100">Schedule Clip</h3>
              <button @click="scheduleModal = false" class="text-gray-400 hover:text-gray-200">&times;</button>
            </div>
            <div class="text-sm text-gray-400">Date: {{ selectedDate }}</div>

            <div v-if="availableClips.length === 0" class="text-gray-500 text-sm py-4 text-center">
              No selected clips available to schedule.
            </div>
            <div v-else class="space-y-3">
              <div>
                <label class="block text-xs text-gray-400 mb-1">Clip</label>
                <select v-model="selectedClipId"
                  class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-blue-500">
                  <option v-for="clip in availableClips" :key="clip.clip_id" :value="clip.clip_id">
                    {{ clip.speaker || clip.clip_id }} — {{ clip.hook_text ? clip.hook_text.substring(0, 50) : clip.clip_id }}
                  </option>
                </select>
              </div>
              <div>
                <label class="block text-xs text-gray-400 mb-1">Platform</label>
                <select v-model="selectedPlatform"
                  class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-blue-500">
                  <option v-for="p in platforms" :key="p" :value="p">{{ p }}</option>
                </select>
              </div>
              <div class="flex justify-end gap-2 pt-2">
                <button @click="scheduleModal = false"
                  class="px-3 py-1.5 rounded-lg text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors">
                  Cancel
                </button>
                <button @click="submitSchedule"
                  class="px-3 py-1.5 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors">
                  Schedule
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `
};

// Utility: format a timestamp string to relative time (e.g., "2 minutes ago")
function relativeTime(isoString) {
  if (!isoString) return '';
  const now = Date.now();
  const then = new Date(isoString).getTime();
  const diffSec = Math.floor((now - then) / 1000);
  if (diffSec < 5) return 'just now';
  if (diffSec < 60) return `${diffSec} seconds ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr} hour${diffHr === 1 ? '' : 's'} ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay} day${diffDay === 1 ? '' : 's'} ago`;
}

// Task status sort priority: running first, then queued, then completed/failed
const TASK_STATUS_ORDER = { running: 0, queued: 1, completed: 2, failed: 3 };

// Task status badge styles
const TASK_STATUS_STYLES = {
  queued:    'bg-gray-600 text-gray-200',
  running:   'bg-blue-600 text-blue-100',
  completed: 'bg-green-600 text-green-100',
  failed:    'bg-red-600 text-red-100',
};

// Tasks View — real-time monitoring of background reprocessing jobs
const TasksView = {
  emits: ['task-count'],
  setup(props, { emit }) {
    const tasks = ref([]);
    let eventSource = null;
    let pollInterval = null;

    const sortedTasks = computed(() => {
      return [...tasks.value].sort((a, b) => {
        const orderA = TASK_STATUS_ORDER[a.status] ?? 99;
        const orderB = TASK_STATUS_ORDER[b.status] ?? 99;
        if (orderA !== orderB) return orderA - orderB;
        // Within same priority group, newest first
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
    });

    const activeCount = computed(() => {
      return tasks.value.filter(t => t.status === 'running' || t.status === 'queued').length;
    });

    watch(activeCount, (count) => {
      emit('task-count', count);
    });

    function updateTasks(data) {
      tasks.value = data;
    }

    function startSSE() {
      eventSource = new EventSource('/api/tasks/stream');
      eventSource.onmessage = (event) => {
        try {
          updateTasks(JSON.parse(event.data));
        } catch (e) {
          console.error('Failed to parse SSE data:', e);
        }
      };
      eventSource.onerror = () => {
        // SSE failed, fall back to polling
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
        startPolling();
      };
    }

    async function fetchTasks() {
      try {
        const resp = await fetch('/api/tasks');
        if (resp.ok) {
          updateTasks(await resp.json());
        }
      } catch (e) {
        console.error('Failed to fetch tasks:', e);
      }
    }

    function startPolling() {
      if (pollInterval) return;
      fetchTasks();
      pollInterval = setInterval(fetchTasks, 2000);
    }

    function cleanup() {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }

    onMounted(() => {
      startSSE();
    });

    onUnmounted(() => {
      cleanup();
    });

    function taskStatusClass(status) {
      return TASK_STATUS_STYLES[status] || TASK_STATUS_STYLES.queued;
    }

    return { sortedTasks, activeCount, relativeTime, taskStatusClass };
  },
  template: `
    <div>
      <!-- Empty state -->
      <div v-if="sortedTasks.length === 0" class="text-gray-500 text-center py-20">
        <p class="text-lg">No background tasks</p>
      </div>

      <!-- Task cards -->
      <div v-else class="space-y-3">
        <div v-for="task in sortedTasks" :key="task.task_id"
          class="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-2">

          <!-- Header row: type + target, status badge -->
          <div class="flex items-center justify-between gap-3">
            <div class="text-sm font-medium text-gray-200 truncate">
              <span class="capitalize">{{ task.task_type }}</span><span v-if="task.target">: {{ task.target }}</span>
            </div>
            <span :class="['px-2 py-0.5 rounded-full text-xs font-medium capitalize flex-shrink-0',
              taskStatusClass(task.status)]"
              v-text="task.status"></span>
          </div>

          <!-- Description -->
          <p v-if="task.description" class="text-xs text-gray-400">{{ task.description }}</p>

          <!-- Progress bar for running tasks -->
          <div v-if="task.status === 'running' && task.progress != null"
            class="w-full bg-gray-800 rounded h-2 overflow-hidden">
            <div class="bg-blue-500 h-2 rounded transition-all duration-300"
              :style="{ width: task.progress + '%' }"></div>
          </div>

          <!-- Timestamps and error -->
          <div class="flex items-center gap-3 text-xs text-gray-500">
            <span v-if="task.created_at">Started {{ relativeTime(task.created_at) }}</span>
            <span v-if="task.completed_at">Finished {{ relativeTime(task.completed_at) }}</span>
          </div>

          <!-- Error message -->
          <p v-if="task.error" class="text-xs text-red-400">{{ task.error }}</p>
        </div>
      </div>
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
