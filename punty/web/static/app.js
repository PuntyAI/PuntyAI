/**
 * PuntyAI - Dashboard JavaScript
 */

// HTMX configuration
document.body.addEventListener('htmx:configRequest', function(evt) {
    // Add CSRF token if needed
    // evt.detail.headers['X-CSRF-Token'] = getCSRFToken();
});

// Handle successful responses
document.body.addEventListener('htmx:afterRequest', function(evt) {
    if (evt.detail.successful) {
        // Check for toast messages in response headers
        const message = evt.detail.xhr.getResponseHeader('X-Toast-Message');
        const type = evt.detail.xhr.getResponseHeader('X-Toast-Type') || 'success';
        if (message) {
            showToast(message, type);
        }
    }
});

// Handle errors
document.body.addEventListener('htmx:responseError', function(evt) {
    const message = evt.detail.xhr.responseText || 'An error occurred';
    showToast(message, 'error');
});

// SSE event source for real-time updates
let eventSource = null;

function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/api/events');

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleSSEEvent(data);
    };

    eventSource.onerror = function(err) {
        console.error('SSE error:', err);
        // Reconnect after 5 seconds
        setTimeout(connectSSE, 5000);
    };
}

function handleSSEEvent(data) {
    switch (data.type) {
        case 'content_generated':
            showToast(`New content generated: ${data.content_type}`, 'success');
            // Refresh content lists
            htmx.trigger('#content-list', 'refresh');
            htmx.trigger('#review-badge', 'refresh');
            break;

        case 'data_updated':
            showToast(`Data updated for ${data.meeting_id}`, 'info');
            break;

        case 'speed_maps_available':
            showToast(`Speed maps available: ${data.meeting_id}`, 'warning');
            break;

        default:
            console.log('Unknown SSE event:', data);
    }
}

// Connect SSE when page loads
// Uncomment when SSE endpoint is implemented
// document.addEventListener('DOMContentLoaded', connectSSE);

// Character counter for Twitter
function updateCharCount(textarea, counterId) {
    const count = textarea.value.length;
    const counter = document.getElementById(counterId);
    if (counter) {
        counter.textContent = count;
        counter.classList.toggle('text-red-500', count > 280);
    }
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(err => {
        showToast('Failed to copy', 'error');
    });
}

// Format timestamp
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('en-AU', {
        day: 'numeric',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit'
    });
}
