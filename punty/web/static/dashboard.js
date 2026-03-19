/**
 * PuntyAI - Dashboard JavaScript
 */

// --- Wins Ticker ---
(function() {
    function loadWinsTicker() {
        fetch('/api/results/wins/recent?limit=15')
            .then(function(resp) { return resp.json(); })
            .then(function(data) {
                var ticker = document.getElementById('ticker-content');
                if (!ticker) return;

                if (!data.wins || data.wins.length === 0) {
                    ticker.innerHTML = '<span class="text-gray-500 text-sm px-4">No wins yet today - let\'s change that!</span>';
                    ticker.style.animation = 'none';
                    return;
                }

                var items = '';
                for (var i = 0; i < 2; i++) {
                    data.wins.forEach(function(win) {
                        items += '<div class="win-item">' +
                            '<span class="win-celebration">' + win.celebration + '</span>' +
                            '<span class="win-horse">' + win.display_name + '</span>' +
                            '<span class="win-venue">' + win.venue + '</span>' +
                            '<span class="win-amounts">' +
                                '<span class="win-stake">$' + win.stake.toFixed(0) + '</span>' +
                                ' \u2192 ' +
                                '<span class="win-return">$' + win.returned.toFixed(0) + '</span>' +
                            '</span>' +
                        '</div>';
                    });
                }
                ticker.innerHTML = items;

                var totalWidth = ticker.scrollWidth / 2;
                var speed = Math.max(10, totalWidth / 120);
                ticker.style.animation = 'ticker-scroll ' + speed + 's linear infinite';

                // Pause ticker when tab is hidden
                document.addEventListener('visibilitychange', function() {
                    if (document.hidden) {
                        ticker.style.animationPlayState = 'paused';
                    } else {
                        ticker.style.animationPlayState = 'running';
                    }
                });
            })
            .catch(function(e) {
                console.error('Failed to load wins ticker:', e);
            });
    }

    loadWinsTicker();
    setInterval(loadWinsTicker, 60000);
})();

// --- Monitor Toggle ---
function initMonitor(initialRunning) {
    var monitorRunning = initialRunning;

    window.toggleMonitor = function() {
        var btn = document.getElementById('monitor-toggle-btn');
        var action = monitorRunning ? 'stop' : 'start';
        btn.disabled = true;
        btn.classList.add('opacity-50');
        btn.textContent = action === 'start' ? 'Starting...' : 'Stopping...';

        fetch('/api/results/monitor/' + action, { method: 'POST' })
            .then(function(r) { return r.json(); })
            .then(function() {
                monitorRunning = !monitorRunning;
                updateMonitorUI(monitorRunning);
                btn.disabled = false;
                btn.classList.remove('opacity-50');
            })
            .catch(function() {
                btn.disabled = false;
                btn.classList.remove('opacity-50');
                updateMonitorUI(monitorRunning);
            });
    };

    function updateMonitorUI(running) {
        var btn = document.getElementById('monitor-toggle-btn');
        var statusDot = document.getElementById('monitor-status-dot');
        var statusText = document.getElementById('monitor-status-text');

        btn.textContent = running ? 'Stop' : 'Start';
        btn.className = (running
            ? 'bg-red-100 text-red-700 border-red-300 hover:bg-red-200'
            : 'bg-green-100 text-green-700 border-green-300 hover:bg-green-200')
            + ' px-4 py-2 rounded-lg border font-medium text-sm';

        if (statusDot) statusDot.textContent = running ? '\u{1F7E2}' : '\u26AA';
        if (statusText) {
            statusText.textContent = running ? 'Running' : 'Stopped';
            statusText.className = 'text-lg font-bold ' + (running ? 'text-green-700' : 'text-gray-500');
        }
    }
}

// --- Countdown timers for meetings not started ---
(function() {
    function updateCountdowns() {
        var timers = document.querySelectorAll('.countdown-timer');
        var now = new Date();
        timers.forEach(function(el) {
            var startTime = el.getAttribute('data-start-time');
            if (!startTime) return;
            var start = new Date(startTime);
            var diff = start - now;
            if (diff <= 0) {
                el.textContent = 'Starting now';
                el.classList.remove('bg-amber-100', 'text-amber-900');
                el.classList.add('bg-blue-100', 'text-blue-700');
            } else {
                var hours = Math.floor(diff / (1000 * 60 * 60));
                var minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                if (hours > 0) {
                    el.textContent = hours + 'h ' + minutes + 'm';
                } else {
                    el.textContent = minutes + 'm';
                }
            }
        });
    }
    updateCountdowns();
    setInterval(updateCountdowns, 60000);
})();

// --- Activity Log ---
(function() {
    var statusIcons = {success: '\u2713', error: '\u2717', warning: '\u26A0', info: '\u25CB'};
    var statusColors = {
        success: 'text-green-500',
        error: 'text-red-500',
        warning: 'text-yellow-500',
        info: 'text-gray-400'
    };

    function refreshDashActivityLog() {
        fetch('/api/scheduler/full-status')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                var badge = document.getElementById('dash-scrape-status');
                if (badge) {
                    if (data.scrape_in_progress) {
                        badge.textContent = 'Scraping: ' + (data.current_scrape || '...');
                        badge.className = 'text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 animate-pulse';
                    } else {
                        badge.textContent = 'Idle';
                        badge.className = 'text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500';
                    }
                }

                var jobsEl = document.getElementById('dash-pending-jobs');
                if (jobsEl && data.pending_jobs) {
                    if (data.pending_jobs.length === 0) {
                        jobsEl.innerHTML = '<span class="text-xs text-gray-400">No pending jobs</span>';
                    } else {
                        var html = '';
                        data.pending_jobs.forEach(function(job) {
                            var name = job.name || job.id || '?';
                            name = name.replace(/-2026-\d{2}-\d{2}/, '').replace('setup_meeting_automation.<locals>.', '');
                            var time = job.next_run || '';
                            html += '<span class="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">' +
                                '<span class="font-medium">' + name + '</span>' +
                                (time ? '<span class="text-indigo-400">' + time + '</span>' : '') +
                                '</span>';
                        });
                        jobsEl.innerHTML = html;
                    }
                }

                var logEl = document.getElementById('dash-activity-log');
                if (logEl && data.activity_log) {
                    if (data.activity_log.length === 0) {
                        logEl.innerHTML = '<span class="text-gray-400">No activity yet</span>';
                    } else {
                        var html = '';
                        data.activity_log.forEach(function(entry) {
                            var icon = statusIcons[entry.status] || '\u25CB';
                            var colorClass = statusColors[entry.status] || 'text-gray-400';
                            var time = entry.time_str || '';
                            var msg = entry.message || '';
                            var venue = entry.venue ? ' <span class="text-gray-500">[' + entry.venue + ']</span>' : '';
                            html += '<div class="flex items-start gap-2 py-0.5">' +
                                '<span class="text-gray-500 shrink-0 w-12">' + time + '</span>' +
                                '<span class="' + colorClass + ' shrink-0">' + icon + '</span>' +
                                '<span class="text-gray-300">' + msg + venue + '</span>' +
                                '</div>';
                        });
                        logEl.innerHTML = html;
                    }
                }
            })
            .catch(function(e) { console.error('Activity log fetch failed:', e); });
    }

    refreshDashActivityLog();
    setInterval(refreshDashActivityLog, 10000);
})();

// --- P&L Chart (lazy-loaded) ---
function initPnlChart(rawData) {
    if (!rawData || rawData.length === 0) return;

    function loadChartAndRender() {
        var script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';
        script.onload = function() { renderChart(); };
        document.head.appendChild(script);
    }

    function renderChart() {
        var labels = rawData.map(function(d) { return d.label || d.date; });

        function pct(pnl, staked) { return staked ? (pnl / staked * 100) : 0; }
        var totalValues = rawData.map(function(d) { return pct(d.cumulative_pnl, d.cumulative_staked); });
        var selectionValues = rawData.map(function(d) { return pct(d.cumulative_selection || 0, d.cumulative_staked); });
        var exoticValues = rawData.map(function(d) { return pct(d.cumulative_exotic || 0, d.cumulative_staked); });
        var sequenceValues = rawData.map(function(d) { return pct(d.cumulative_sequence || 0, d.cumulative_staked); });
        var big3Values = rawData.map(function(d) { return pct(d.cumulative_big3_multi || 0, d.cumulative_staked); });

        var finalVal = totalValues[totalValues.length - 1];
        var totalColor = finalVal >= 0 ? '#4ade80' : '#f87171';
        var totalFill = finalVal >= 0 ? 'rgba(74,222,128,0.08)' : 'rgba(248,113,113,0.08)';

        var pointRadius = rawData.length <= 30 ? 3 : rawData.length <= 60 ? 2 : 0;

        var ctx = document.getElementById('pnlChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Total', data: totalValues,
                        borderColor: totalColor, backgroundColor: totalFill,
                        borderWidth: 3, fill: true, tension: 0.3,
                        pointRadius: pointRadius, pointHoverRadius: 6, order: 0,
                    },
                    {
                        label: 'Selections', data: selectionValues,
                        borderColor: '#f472b6', backgroundColor: 'transparent',
                        borderWidth: 2, fill: false, tension: 0.3,
                        pointRadius: 0, pointHoverRadius: 4, order: 1,
                    },
                    {
                        label: 'Exotics', data: exoticValues,
                        borderColor: '#38bdf8', backgroundColor: 'transparent',
                        borderWidth: 2, fill: false, tension: 0.3,
                        pointRadius: 0, pointHoverRadius: 4, order: 2,
                    },
                    {
                        label: 'Sequences', data: sequenceValues,
                        borderColor: '#facc15', backgroundColor: 'transparent',
                        borderWidth: 2, fill: false, tension: 0.3,
                        pointRadius: 0, pointHoverRadius: 4, order: 3,
                    },
                    {
                        label: 'Big 3 Multi', data: big3Values,
                        borderColor: '#a78bfa', backgroundColor: 'transparent',
                        borderWidth: 2, fill: false, tension: 0.3,
                        pointRadius: 0, pointHoverRadius: 4, order: 4,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        display: true, position: 'top',
                        labels: {
                            color: '#9ca3af', usePointStyle: true, pointStyle: 'line',
                            padding: 15, font: { family: 'Rajdhani', size: 12 }
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1a1a25',
                        borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
                        titleFont: { family: 'Rajdhani', size: 14, weight: '600' },
                        bodyFont: { family: 'Source Sans Pro', size: 12 },
                        padding: 12,
                        callbacks: {
                            title: function(items) {
                                var d = rawData[items[0].dataIndex];
                                var periodLabel = {hour: 'Hour', day: 'Day', week: 'Week', start: 'Start'}[d.period_type] || 'Period';
                                return d.label + ' (' + periodLabel + ')';
                            },
                            label: function(ctx) {
                                var val = ctx.parsed.y;
                                return ctx.dataset.label + ': ' + (val >= 0 ? '+' : '') + val.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#606070', font: { family: 'Rajdhani', size: 12 } },
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: {
                            color: '#606070',
                            font: { family: 'Rajdhani', size: 12 },
                            callback: function(v) { return (v >= 0 ? '+' : '') + v.toFixed(1) + '%'; }
                        },
                    }
                }
            }
        });
    }

    // Use IntersectionObserver to lazy-load chart when visible
    var chartCanvas = document.getElementById('pnlChart');
    if (!chartCanvas) return;

    if ('IntersectionObserver' in window) {
        var observer = new IntersectionObserver(function(entries) {
            if (entries[0].isIntersecting) {
                observer.disconnect();
                loadChartAndRender();
            }
        }, { rootMargin: '200px' });
        observer.observe(chartCanvas);
    } else {
        loadChartAndRender();
    }
}
