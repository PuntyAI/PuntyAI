/**
 * PuntyAI - Meets page JavaScript
 */

// --- Selection management ---
function updateSelectedCount() {
    var toggles = document.querySelectorAll('[data-selected="true"]');
    var el = document.getElementById('selected-count');
    if (el) el.textContent = toggles.length + ' selected';
}

async function toggleSelect(meetingId, btn) {
    try {
        var resp = await fetch('/api/meets/' + meetingId + '/select', { method: 'PUT' });
        var data = await resp.json();
        var selected = data.selected;
        btn.dataset.selected = selected ? 'true' : 'false';
        btn.className = btn.className.replace(
            selected ? 'bg-gray-300' : 'bg-punty-coral',
            selected ? 'bg-punty-coral' : 'bg-gray-300'
        );
        btn.setAttribute('aria-checked', selected ? 'true' : 'false');
        var dot = btn.querySelector('span');
        dot.className = dot.className.replace(
            selected ? 'translate-x-1' : 'translate-x-6',
            selected ? 'translate-x-6' : 'translate-x-1'
        );
        updateSelectedCount();
        updateActiveAllToggle();
        // Dim/brighten the row for past meetings
        var row = btn.closest('tr[data-meeting-id]');
        if (row) {
            row.classList.toggle('opacity-50', !selected);
        }
    } catch (e) {
        showToast('Failed to toggle selection', 'error');
    }
}

// --- Calendar scrape ---
function scrapeCalendar() {
    var btn = document.getElementById('btn-refresh');
    var progressDiv = document.getElementById('scrape-progress');
    var progressBar = document.getElementById('scrape-progress-bar');
    var progressLabel = document.getElementById('scrape-progress-label');
    var progressPct = document.getElementById('scrape-progress-pct');
    var progressDetail = document.getElementById('scrape-progress-detail');
    var spinner = document.getElementById('scrape-spinner');
    var resultsDiv = document.getElementById('scrape-results');
    var resultsContent = document.getElementById('scrape-results-content');

    btn.disabled = true;
    btn.querySelector('span').textContent = 'Scraping...';

    progressDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    progressBar.style.width = '100%';
    progressBar.classList.add('animate-pulse');
    progressBar.classList.remove('bg-green-500', 'bg-red-500');
    progressBar.classList.add('bg-punty-coral');
    spinner.classList.remove('hidden');
    progressLabel.textContent = 'Fetching calendar from racing.com...';
    progressPct.textContent = '';
    progressDetail.textContent = 'This can take 1-3 minutes due to page rendering';
    progressDetail.className = 'text-xs text-gray-500';

    fetch('/api/meets/scrape-calendar', { method: 'POST' })
        .then(function(resp) {
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
        .then(function(data) {
            progressBar.classList.remove('animate-pulse', 'bg-punty-coral');
            progressBar.classList.add('bg-green-500');
            spinner.classList.add('hidden');
            progressLabel.textContent = '\u2713 Calendar scraped successfully';
            progressPct.textContent = '';
            progressDetail.textContent = '';

            resultsDiv.classList.remove('hidden');
            resultsDiv.className = 'mt-2 p-3 rounded-lg text-sm bg-green-50 border border-green-200';
            resultsContent.innerHTML =
                '<div class="font-medium text-green-800 mb-1">Found ' + data.count + ' meetings</div>' +
                '<div class="text-green-700">Page will reload in 2 seconds...</div>';

            setTimeout(function() { location.reload(); }, 2000);
        })
        .catch(function(e) {
            progressBar.classList.remove('animate-pulse', 'bg-punty-coral');
            progressBar.classList.add('bg-red-500');
            spinner.classList.add('hidden');
            progressLabel.textContent = '\u2717 Calendar scrape failed';
            progressPct.textContent = '';

            resultsDiv.classList.remove('hidden');
            resultsDiv.className = 'mt-2 p-3 rounded-lg text-sm bg-red-50 border border-red-200';
            resultsContent.innerHTML =
                '<div class="font-medium text-red-800 mb-1">Error: ' + e.message + '</div>' +
                '<div class="text-red-700">Check server logs or try again</div>';
        })
        .finally(function() {
            btn.disabled = false;
            btn.querySelector('span').textContent = 'Refresh Calendar';
        });
}

// --- Single meeting scrape with EventSource + timeout ---
var EVENTSOURCE_TIMEOUT_MS = 300000; // 5 minutes

function scrapeFull(meetingId) {
    var progressDiv = document.getElementById('scrape-progress');
    var progressBar = document.getElementById('scrape-progress-bar');
    var progressLabel = document.getElementById('scrape-progress-label');
    var progressPct = document.getElementById('scrape-progress-pct');
    var progressDetail = document.getElementById('scrape-progress-detail');
    var spinner = document.getElementById('scrape-spinner');
    var resultsDiv = document.getElementById('scrape-results');
    var resultsContent = document.getElementById('scrape-results-content');

    progressDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    progressBar.style.width = '0%';
    progressBar.classList.remove('animate-pulse', 'bg-green-500', 'bg-red-500', 'bg-yellow-500');
    progressBar.classList.add('bg-punty-coral');
    spinner.classList.remove('hidden');
    progressLabel.textContent = 'Connecting...';
    progressPct.textContent = '0%';
    progressDetail.textContent = '';
    progressDetail.className = 'text-xs text-gray-500';

    var stepResults = [];

    var eventSource = new EventSource('/api/meets/' + meetingId + '/scrape-stream');
    var lastStep = -1;

    // Timeout guard
    var timeoutId = setTimeout(function() {
        eventSource.close();
        spinner.classList.add('hidden');
        progressBar.classList.remove('bg-punty-coral');
        progressBar.classList.add('bg-yellow-500');
        progressLabel.textContent = '\u26A0 Timed out';
        resultsDiv.classList.remove('hidden');
        resultsDiv.className = 'mt-2 p-3 rounded-lg text-sm bg-yellow-50 border border-yellow-200';
        resultsContent.innerHTML =
            '<div class="font-medium text-yellow-800 mb-1">Scrape timed out after 5 minutes</div>' +
            '<div class="text-yellow-700">The server may still be processing. Check the meeting page or try again.</div>';
    }, EVENTSOURCE_TIMEOUT_MS);

    eventSource.onmessage = function(event) {
        var data = JSON.parse(event.data);
        var step = data.step || 0;
        var total = data.total || 3;
        var pct = Math.round((step / total) * 100);

        progressBar.style.width = pct + '%';
        progressPct.textContent = pct + '%';
        progressLabel.textContent = data.label || 'Scraping...';

        if (step !== lastStep && data.status !== 'running') {
            stepResults.push({ step: step, label: data.label, status: data.status });
            lastStep = step;
        }

        if (data.status === 'error') {
            progressDetail.innerHTML = '<span class="text-red-600">\u26A0 ' + data.label + '</span>';
        } else if (data.status === 'done') {
            progressDetail.innerHTML = '<span class="text-green-600">\u2713 Step ' + step + ' complete</span>';
        } else if (data.status === 'running') {
            progressDetail.innerHTML = '<span class="text-gray-500">Processing...</span>';
        }

        if (data.status === 'complete') {
            clearTimeout(timeoutId);
            eventSource.close();
            spinner.classList.add('hidden');

            var hasErrors = data.errors && data.errors.length > 0;

            progressBar.classList.remove('bg-punty-coral');
            if (hasErrors) {
                progressBar.classList.add('bg-yellow-500');
                progressLabel.textContent = '\u26A0 Completed with errors';
            } else {
                progressBar.classList.add('bg-green-500');
                progressLabel.textContent = '\u2713 Scrape complete';
            }
            progressBar.style.width = '100%';
            progressPct.textContent = '100%';
            progressDetail.textContent = '';

            resultsDiv.classList.remove('hidden');
            resultsDiv.className = hasErrors
                ? 'mt-2 p-3 rounded-lg text-sm bg-yellow-50 border border-yellow-200'
                : 'mt-2 p-3 rounded-lg text-sm bg-green-50 border border-green-200';

            var summaryHtml = '<div class="space-y-1">';
            for (var i = 0; i < stepResults.length; i++) {
                var sr = stepResults[i];
                var icon = sr.status === 'error' ? '\u2717' : '\u2713';
                var color = sr.status === 'error' ? 'text-red-600' : 'text-green-700';
                summaryHtml += '<div class="' + color + '">' + icon + ' ' + sr.label + '</div>';
            }
            if (hasErrors) {
                summaryHtml += '<div class="text-yellow-700 mt-2 font-medium">Errors: ' + data.errors.join(', ') + '</div>';
            }
            summaryHtml += '<div class="text-gray-600 mt-2">Page will reload in 3 seconds...</div>';
            summaryHtml += '</div>';
            resultsContent.innerHTML = summaryHtml;

            setTimeout(function() {
                progressDiv.classList.add('hidden');
                location.reload();
            }, 3000);
        }
    };

    eventSource.onerror = function() {
        clearTimeout(timeoutId);
        eventSource.close();
        spinner.classList.add('hidden');
        progressBar.classList.remove('bg-punty-coral');
        progressBar.classList.add('bg-red-500');
        progressLabel.textContent = '\u2717 Connection lost';

        resultsDiv.classList.remove('hidden');
        resultsDiv.className = 'mt-2 p-3 rounded-lg text-sm bg-red-50 border border-red-200';
        resultsContent.innerHTML =
            '<div class="font-medium text-red-800 mb-1">Scrape connection lost</div>' +
            '<div class="text-red-700">The server may still be processing. Check the meeting page or try again.</div>';
    };
}

// --- Odds refresh ---
async function refreshOdds(meetingId) {
    try {
        var resp = await fetch('/api/meets/' + meetingId + '/refresh-odds', { method: 'POST' });
        var data = await resp.json();
        showToast(data.status === 'ok' ? 'Odds updated' : 'Odds refresh failed', data.status === 'ok' ? 'success' : 'error');
    } catch (e) {
        showToast('Odds refresh failed', 'error');
    }
}

// --- Quick select ---
async function quickSelect(mode) {
    var rows = document.querySelectorAll('#meetings-body [data-meeting-id]');
    for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        var btn = row.querySelector('[data-selected]');
        var isSelected = btn.dataset.selected === 'true';
        var shouldSelect = false;

        if (mode === 'all') {
            shouldSelect = true;
        } else if (mode === 'none') {
            shouldSelect = false;
        } else if (mode === 'metro') {
            var venue = row.querySelector('a').textContent.trim().toLowerCase();
            var metros = ['flemington', 'caulfield', 'moonee valley', 'sandown',
                          'randwick', 'rosehill', 'warwick farm', 'canterbury',
                          'doomben', 'eagle farm', 'morphettville', 'ascot'];
            shouldSelect = metros.some(function(m) { return venue.includes(m); });
        } else {
            var venue = row.querySelector('a').textContent.trim().toLowerCase();
            var stateMap = {
                'VIC': ['flemington', 'caulfield', 'moonee valley', 'sandown', 'cranbourne', 'pakenham'],
                'NSW': ['randwick', 'rosehill', 'warwick farm', 'canterbury', 'newcastle', 'kembla'],
                'QLD': ['doomben', 'eagle farm', 'gold coast', 'sunshine coast'],
            };
            var venues = stateMap[mode] || [];
            shouldSelect = venues.some(function(v) { return venue.includes(v); });
        }

        if (shouldSelect !== isSelected) {
            await toggleSelect(row.dataset.meetingId, btn);
        }
    }
}

// --- Bulk Actions ---
var bulkEventSource = null;

function setBulkButtonsDisabled(disabled) {
    ['btn-bulk-scrape', 'btn-bulk-generate'].forEach(function(id) {
        var btn = document.getElementById(id);
        if (btn) {
            btn.disabled = disabled;
            if (disabled) {
                btn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                btn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
    });
    var cancelBtn = document.getElementById('btn-bulk-cancel');
    if (cancelBtn) {
        if (disabled) {
            cancelBtn.classList.remove('hidden');
        } else {
            cancelBtn.classList.add('hidden');
        }
    }
}

function cancelBulk() {
    if (bulkEventSource) {
        bulkEventSource.close();
        bulkEventSource = null;
    }
    setBulkButtonsDisabled(false);
    var bar = document.getElementById('bulk-progress-bar');
    bar.classList.remove('bg-punty-coral', 'bg-punty-blue', 'bg-punty-gold');
    bar.classList.add('bg-yellow-500');
    var spinner = document.getElementById('bulk-spinner');
    spinner.classList.add('hidden');
    document.getElementById('bulk-progress-label').textContent = 'Cancelled';
    addBulkLog('Cancelled by user', 'warning');
    showToast('Bulk operation cancelled', 'warning');
}

function addBulkLog(message, type) {
    var log = document.getElementById('bulk-log');
    var line = document.createElement('div');
    line.className = 'flex items-center gap-2 py-0.5';
    var icon = '';
    var textClass = 'text-gray-600';
    if (type === 'running') {
        icon = '<span class="inline-block w-3 h-3 border-2 border-punty-coral border-t-transparent rounded-full animate-spin"></span>';
    } else if (type === 'done') {
        icon = '<span class="text-green-600">&#10003;</span>';
        textClass = 'text-green-700';
    } else if (type === 'error') {
        icon = '<span class="text-red-600">&#10007;</span>';
        textClass = 'text-red-600';
    } else if (type === 'warning') {
        icon = '<span class="text-yellow-500">&#x26A0;</span>';
        textClass = 'text-yellow-600';
    } else if (type === 'complete') {
        icon = '<span class="text-punty-coral">&#9679;</span>';
        textClass = 'font-semibold text-gray-900';
    }
    line.innerHTML = icon + '<span class="' + textClass + '">' + message + '</span>';
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

function runBulkStream(endpoint, barColor, actionName) {
    var panel = document.getElementById('bulk-progress');
    var bar = document.getElementById('bulk-progress-bar');
    var label = document.getElementById('bulk-progress-label');
    var pct = document.getElementById('bulk-progress-pct');
    var detail = document.getElementById('bulk-progress-detail');
    var spinner = document.getElementById('bulk-spinner');
    var log = document.getElementById('bulk-log');

    panel.classList.remove('hidden');
    bar.style.width = '0%';
    bar.className = barColor + ' h-3 rounded-full transition-all duration-300';
    spinner.classList.remove('hidden');
    label.textContent = 'Starting ' + actionName + '...';
    pct.textContent = '0/0';
    detail.textContent = '';
    log.innerHTML = '';

    setBulkButtonsDisabled(true);

    bulkEventSource = new EventSource(endpoint);

    // Timeout guard for bulk operations (10 minutes)
    var bulkTimeoutId = setTimeout(function() {
        if (bulkEventSource) {
            bulkEventSource.close();
            bulkEventSource = null;
        }
        setBulkButtonsDisabled(false);
        spinner.classList.add('hidden');
        bar.classList.add('bg-yellow-500');
        label.textContent = '\u26A0 Timed out';
        addBulkLog('Operation timed out after 10 minutes', 'warning');
        showToast('Bulk operation timed out', 'warning');
    }, 600000);

    bulkEventSource.onmessage = function(e) {
        var data = JSON.parse(e.data);
        var step = data.step || 0;
        var total = data.total_meetings || data.total || 1;
        var percent = total > 0 ? Math.round((step / total) * 100) : 0;

        bar.style.width = percent + '%';
        pct.textContent = step + '/' + total;
        label.textContent = data.label || 'Processing...';

        if (data.meeting) {
            detail.textContent = 'Current: ' + data.meeting;
        }

        if (data.status === 'done' && data.meeting) {
            addBulkLog(data.label || data.meeting + ' complete', 'done');
        } else if (data.status === 'error') {
            addBulkLog(data.label || 'Error', 'error');
        } else if (data.status === 'complete') {
            clearTimeout(bulkTimeoutId);
            bulkEventSource.close();
            bulkEventSource = null;
            setBulkButtonsDisabled(false);
            spinner.classList.add('hidden');
            bar.classList.remove('bg-punty-coral', 'bg-punty-blue', 'bg-punty-gold');
            bar.classList.add('bg-green-500');
            bar.style.width = '100%';
            addBulkLog(data.label || 'Complete', 'complete');
            showToast(actionName + ' complete!', 'success');
            setTimeout(function() { location.reload(); }, 2000);
        }
    };

    bulkEventSource.onerror = function() {
        clearTimeout(bulkTimeoutId);
        bulkEventSource.close();
        bulkEventSource = null;
        setBulkButtonsDisabled(false);
        spinner.classList.add('hidden');
        bar.classList.remove('bg-punty-coral', 'bg-punty-blue', 'bg-punty-gold');
        bar.classList.add('bg-red-500');
        label.textContent = 'Connection lost';
        addBulkLog('Connection lost - check server logs', 'error');
        showToast('Connection lost', 'error');
    };
}

function bulkScrapeAndSpeedMaps() {
    runBulkStream('/api/meets/bulk/scrape-and-speedmaps-stream', 'bg-punty-coral', 'Scrape All Data');
}

async function bulkGenerate() {
    var response = await fetch('/api/meets/incomplete-data-check');
    var data = await response.json();

    if (data.incomplete_meetings && data.incomplete_meetings.length > 0) {
        var venues = data.incomplete_meetings.map(function(m) { return m.venue; }).join(', ');
        var proceed = await puntyConfirm(
            'Data Warning',
            'The following meetings have incomplete or missing speed map data:\n\n' +
            venues + '\n\n' +
            'Generating Early Mail with incomplete data may result in missing pace analysis and less accurate selections.',
            'Proceed Anyway',
            'Cancel'
        );
        if (!proceed) {
            showToast('Generation cancelled', 'warning');
            return;
        }
    }

    runBulkStream('/api/meets/bulk/generate-early-mail-stream', 'bg-punty-gold', 'Generate Early Mail');
}

// --- Active All toggle ---
async function toggleActiveAll(btn) {
    var toggleEl = document.getElementById('active-all-toggle');
    var allToggles = document.querySelectorAll('[data-selected]');
    var allSelected = Array.from(allToggles).every(function(t) { return t.dataset.selected === 'true'; });
    var activate = !allSelected;

    btn.disabled = true;
    btn.classList.add('opacity-50');

    try {
        var resp = await fetch('/api/meets/bulk/select-all?activate=' + activate, { method: 'PUT' });
        var data = await resp.json();

        if (data.status === 'ok') {
            allToggles.forEach(function(toggle) {
                toggle.dataset.selected = activate ? 'true' : 'false';
                toggle.setAttribute('aria-checked', activate ? 'true' : 'false');
                toggle.className = toggle.className.replace(
                    activate ? 'bg-gray-300' : 'bg-punty-coral',
                    activate ? 'bg-punty-coral' : 'bg-gray-300'
                );
                var dot = toggle.querySelector('span');
                dot.className = dot.className.replace(
                    activate ? 'translate-x-1' : 'translate-x-6',
                    activate ? 'translate-x-6' : 'translate-x-1'
                );
            });

            updateActiveAllToggle();
            updateSelectedCount();
            showToast(data.count + ' meetings ' + (activate ? 'activated' : 'deactivated'), 'success');
        }
    } catch (e) {
        showToast('Failed to toggle all meetings', 'error');
    } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-50');
    }
}

function updateActiveAllToggle() {
    var toggleEl = document.getElementById('active-all-toggle');
    var dotEl = document.getElementById('active-all-dot');
    var labelEl = document.getElementById('active-all-label');

    var allToggles = document.querySelectorAll('[data-selected]');
    var selectedCount = Array.from(allToggles).filter(function(t) { return t.dataset.selected === 'true'; }).length;
    var allSelected = selectedCount === allToggles.length && allToggles.length > 0;

    if (allSelected) {
        toggleEl.className = 'relative inline-flex h-4 w-7 items-center rounded-full transition-colors bg-purple-600';
        dotEl.className = 'inline-block h-3 w-3 transform rounded-full bg-white transition-transform translate-x-3.5';
        labelEl.textContent = 'All Active';
    } else {
        toggleEl.className = 'relative inline-flex h-4 w-7 items-center rounded-full transition-colors bg-gray-300';
        dotEl.className = 'inline-block h-3 w-3 transform rounded-full bg-white transition-transform translate-x-0.5';
        labelEl.textContent = 'Active All';
    }
}

// --- Race countdown timer ---
function updateCountdowns() {
    var countdowns = document.querySelectorAll('.race-countdown');
    var now = new Date();

    countdowns.forEach(function(el) {
        var startTime = el.dataset.startTime;
        if (!startTime) return;

        var raceTime = new Date(startTime);
        var diff = raceTime - now;

        if (diff <= 0) {
            el.textContent = 'Started';
            el.classList.remove('text-punty-coral');
            el.classList.add('text-gray-400');
        } else {
            var hours = Math.floor(diff / (1000 * 60 * 60));
            var mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
            var secs = Math.floor((diff % (1000 * 60)) / 1000);

            if (hours > 0) {
                el.textContent = hours + 'h ' + mins + 'm';
            } else if (mins > 0) {
                el.textContent = mins + 'm ' + secs + 's';
            } else {
                el.textContent = secs + 's';
                el.classList.add('text-red-500', 'animate-pulse');
            }
        }
    });
}

// --- Initialize ---
document.addEventListener('DOMContentLoaded', function() {
    updateSelectedCount();
    updateActiveAllToggle();
    updateCountdowns();
    setInterval(updateCountdowns, 1000);
});
