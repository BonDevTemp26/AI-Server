import re

with open('/home/dev-bontech/AI-Server/frontend/theft_videos.html', 'r') as f:
    content = f.read()

# 1. Add CSS
css = """
        .tabs-container { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
        .tab-btn { background: var(--glass-bg); color: var(--text-secondary); border: 1px solid var(--glass-border); padding: 8px 16px; border-radius: 20px; cursor: pointer; transition: all 0.2s; font-family: inherit; font-weight: 500; }
        .tab-btn:hover { background: rgba(168, 85, 247, 0.1); color: #fff; }
        .tab-btn.active { background: #a855f7; color: #fff; border-color: #a855f7; }
        .modal-body-split { display: flex; flex-direction: row; flex-wrap: wrap; }
        .modal-video-section { flex: 2; min-width: 300px; background: #000; }
        .modal-info-section { flex: 1; min-width: 300px; padding: 20px; background: var(--glass-bg); border-left: 1px solid var(--glass-border); display: flex; flex-direction: column; }
        .info-row { margin-bottom: 12px; }
        .info-label { color: var(--text-secondary); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .info-value { color: var(--text-primary); font-size: 1rem; font-weight: 500; }
        .btn-confirm { background: #10b981; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; font-weight: 600; width: 100%; transition: background 0.2s; }
        .btn-confirm:hover { background: #059669; }
        .btn-reject { background: #ef4444; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; font-weight: 600; width: 100%; transition: background 0.2s; }
        .btn-reject:hover { background: #dc2626; }
"""
content = content.replace("</style>", css + "\n    </style>")

# 2. Add Tabs to HTML
tabs_html = """
            <div class="tabs-container" id="filter-tabs">
                <button class="tab-btn active" onclick="setFilter('all')">All</button>
                <button class="tab-btn" onclick="setFilter('pending')">Pending View</button>
                <button class="tab-btn" onclick="setFilter('alert')">Alert</button>
                <button class="tab-btn" onclick="setFilter('confirmed')">Confirm theft</button>
                <button class="tab-btn" onclick="setFilter('false_positive')">False positive</button>
            </div>
"""
content = content.replace('<div class="search-bar" style="margin-top: 20px;">', tabs_html + '\n            <div class="search-bar" style="margin-top: 20px;">')

# 3. Replace Modal HTML
modal_old = """    <div class="video-modal" id="video-modal">
        <div class="video-modal-content">
            <div class="video-modal-header">
                <h3 id="modal-video-title">Camera Name</h3>
                <button class="video-modal-close" onclick="closeVideoModal()">✕</button>
            </div>
            <div class="video-player" id="modal-video-player" style="background:#000;">
                <!-- Video tag injected here -->
            </div>
            <div class="vlm-analysis">
                <div class="vlm-title">
                    <span style="font-size:1.2rem">🤖</span> VLM Analysis Report
                </div>
                <div class="vlm-text">
                    <strong>Action Detected:</strong> Suspicious activity detected aligning with theft profiles.<br><br>
                    <strong>Confidence:</strong> Verified by pipeline.<br><br>
                    <strong>System Notes:</strong> A clip has been captured around the time of detection for manual
                    review.
                </div>
            </div>
        </div>
    </div>"""

modal_new = """    <div class="video-modal" id="video-modal">
        <div class="video-modal-content" style="max-width: 1200px; width: 95%;">
            <div class="video-modal-header">
                <h3 id="modal-video-title">Incident Review</h3>
                <button class="video-modal-close" onclick="closeVideoModal()">✕</button>
            </div>
            <div class="modal-body-split">
                <div class="modal-video-section video-player" id="modal-video-player"></div>
                <div class="modal-info-section">
                    <div class="vlm-title" style="margin-bottom: 20px;">
                        <span style="font-size:1.2rem">🤖</span> Verification Details
                    </div>
                    <div class="info-row">
                        <div class="info-label">Camera Name</div>
                        <div class="info-value" id="modal-cam-name">-</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Detection Score</div>
                        <div class="info-value" id="modal-score">-</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">VLM Reject Score</div>
                        <div class="info-value" id="modal-vlm-score">-</div>
                    </div>
                    <div class="info-row" style="flex: 1;">
                        <div class="info-label">VLM Content</div>
                        <div class="info-value" id="modal-vlm-content" style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.4;">-</div>
                    </div>
                    <div style="display: flex; gap: 10px; margin-top: 20px;">
                        <button class="btn-confirm" onclick="updateReview('confirmed')">Confirm</button>
                        <button class="btn-reject" onclick="updateReview('false_positive')">Reject</button>
                    </div>
                </div>
            </div>
        </div>
    </div>"""
content = content.replace(modal_old, modal_new)

# 4. Replace JS
script_new = """
        let allEvents = [];
        let currentFilter = 'all';
        let currentEventId = null;

        function setFilter(filterType) {
            currentFilter = filterType;
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.innerText.toLowerCase().includes(filterType.replace('_', ' ')) || 
                    (filterType === 'all' && btn.innerText === 'All')) {
                    btn.classList.add('active');
                }
            });
            renderEvents();
        }

        async function loadTheftVideos() {
            try {
                // Fetch recent theft detections
                const res = await fetch('/api/dashboard/detections?type=theft&limit=100');
                allEvents = await res.json();
                renderEvents();
            } catch (e) {
                console.error("Failed to load theft events", e);
                document.getElementById('video-grid').innerHTML = '<div style="color:var(--red); text-align:center; grid-column:1/-1;">Error loading data.</div>';
            }
        }

        function renderEvents() {
            const grid = document.getElementById('video-grid');
            
            const filtered = allEvents.filter(ev => {
                const state = (ev.metadata && ev.metadata.review_state) ? ev.metadata.review_state : 'pending';
                if (currentFilter === 'all') return true;
                return state === currentFilter;
            });

            if (filtered.length === 0) {
                grid.innerHTML = '<div style="color:var(--text-secondary); text-align:center; grid-column:1/-1; padding:40px;">No events found for this filter.</div>';
                return;
            }

            grid.innerHTML = filtered.map(ev => {
                const videoPath = ev.metadata && ev.metadata.video_path ? ev.metadata.video_path : null;
                const date = new Date(ev.detected_at).toLocaleString();
                const shop = ev.shop_name || 'Unknown Shop';
                const cam = ev.camera_name || 'Unknown Camera';
                const confStr = ev.confidence ? (ev.confidence * 100).toFixed(1) + '%' : 'High';
                const state = (ev.metadata && ev.metadata.review_state) ? ev.metadata.review_state : 'pending';
                
                let badgeColor = 'rgba(168, 85, 247, 0.2)';
                let badgeText = '#d8b4fe';
                if(state === 'confirmed') { badgeColor = 'rgba(16, 185, 129, 0.2)'; badgeText = '#34d399'; }
                if(state === 'false_positive') { badgeColor = 'rgba(239, 68, 68, 0.2)'; badgeText = '#fca5a5'; }

                return `
                    <div class="video-card" onclick="openVideoModal(${ev.id})">
                        <div class="video-thumbnail">
                            ${videoPath ? '<div class="play-icon">▶</div>' : '<div style="color:var(--text-secondary)">No Clip Saved</div>'}
                        </div>
                        <div class="video-info">
                            <div class="video-title">
                                ${cam}
                                <span class="severity-badge" style="background:${badgeColor}; color:${badgeText}; border-color:${badgeColor};">${state.toUpperCase()}</span>
                            </div>
                            <div class="video-meta">
                                <span>🏪 ${shop}</span>
                                <span>🕒 ${date}</span>
                                <span>Score: ${confStr}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function openVideoModal(eventId) {
            const ev = allEvents.find(e => e.id === eventId);
            if(!ev) return;
            
            currentEventId = eventId;
            const shop = ev.shop_name || 'Unknown Shop';
            const cam = ev.camera_name || 'Unknown Camera';
            const score = ev.confidence ? (ev.confidence * 100).toFixed(1) + '%' : 'High';
            const vlmRejectScore = ev.metadata && ev.metadata.vlm_reject_score ? (ev.metadata.vlm_reject_score * 100).toFixed(1) + '%' : 'N/A';
            const vlmContent = ev.metadata && ev.metadata.vlm_content ? ev.metadata.vlm_content : 'No VLM description available.';
            const videoPath = ev.metadata && ev.metadata.video_path ? ev.metadata.video_path : null;

            document.getElementById('modal-video-title').innerText = 'Reviewing: ' + shop + ' - ' + cam;
            document.getElementById('modal-cam-name').innerText = cam;
            document.getElementById('modal-score').innerText = score;
            document.getElementById('modal-vlm-score').innerText = vlmRejectScore;
            document.getElementById('modal-vlm-content').innerText = vlmContent;

            const playerContainer = document.getElementById('modal-video-player');
            if (videoPath && videoPath !== 'null') {
                playerContainer.innerHTML = `<video controls autoplay style="width:100%; height:100%; max-height: 60vh;">
                    <source src="${videoPath}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>`;
            } else {
                playerContainer.innerHTML = '<span>[No Video Clip Available for this Event]</span>';
            }

            document.getElementById('video-modal').classList.add('active');
        }

        async function updateReview(state) {
            if(!currentEventId) return;
            try {
                const res = await fetch(`/api/dashboard/detections/${currentEventId}/review`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ review_state: state })
                });
                if(res.ok) {
                    const ev = allEvents.find(e => e.id === currentEventId);
                    if(ev) {
                        if(!ev.metadata) ev.metadata = {};
                        ev.metadata.review_state = state;
                    }
                    renderEvents();
                    closeVideoModal();
                } else {
                    alert("Failed to update status");
                }
            } catch (e) {
                console.error(e);
                alert("Error updating status");
            }
        }
"""
content = re.sub(r'let allEvents = \[\];.*async function updateReview\(state\) \{.*?\}', '', content, flags=re.DOTALL) # In case it was already modified
content = re.sub(r'async function loadTheftVideos\(\).*?\}\);', script_new, content, flags=re.DOTALL)

with open('/home/dev-bontech/AI-Server/frontend/theft_videos.html', 'w') as f:
    f.write(content)
