const API_BASE = '/api';

let oldValues = {}; // Store old values for number animation

async function fetchStats() {
    try {
        const response = await fetch(`${API_BASE}/dashboard/stats`);
        const data = await response.json();

        // Animate numbers for stats
        animateValue('val-shops', data.shops_count);
        animateValue('val-cameras', data.cameras_count);
        animateValue('val-online', data.cameras_online);
        animateValue('val-error', data.cameras_error || 0);

        let theftCount = data.theft_videos_today ?? 0;
        let theftCams = data.theft_cameras_running ?? 0;
        if (data.detailed_detections) {
            const theftData = data.detailed_detections.find(d => (d.type || '').toLowerCase().includes('theft'));
            if (theftData) {
                theftCount = theftData.total_today ?? theftCount;
                theftCams = theftData.cameras_running ?? theftCams;
            }
        } else if (data.detections) {
            theftCount = data.detections['Theft'] ?? data.detections['theft'] ?? theftCount;
        }
        animateValue('val-theft-today', theftCount);
        animateValue('val-theft-cams', theftCams);

        if (data.server_mode === "cloud") {
            const titleOn = document.getElementById('card-online-title');
            if(titleOn) titleOn.innerText = "API Triggers Active";
            const iconOn = document.getElementById('card-online-icon');
            if(iconOn) iconOn.innerText = "☁️";
            const titleOff = document.getElementById('card-offline-title');
            if(titleOff) titleOff.innerText = "Cloud Standby";
            const iconOff = document.getElementById('card-offline-icon');
            if(iconOff) iconOff.innerText = "💤";
            
            // Cloud mode: offline cameras are actually 'standby', so we can show total cameras as standby
            animateValue('val-offline', data.cameras_count);
        } else {
            const titleOn = document.getElementById('card-online-title');
            if(titleOn) titleOn.innerText = "Cameras Online";
            const iconOn = document.getElementById('card-online-icon');
            if(iconOn) iconOn.innerText = "🟢";
            const titleOff = document.getElementById('card-offline-title');
            if(titleOff) titleOff.innerText = "Cameras Offline";
            const iconOff = document.getElementById('card-offline-icon');
            if(iconOff) iconOff.innerText = "❌";
            animateValue('val-offline', data.cameras_offline);
        }
        // Pulse the error card if there are errors
        const errCard = document.getElementById('card-error');
        if (errCard) {
            if ((data.cameras_error || 0) > 0) {
                errCard.style.borderColor = 'rgba(251,146,60,.7)';
                errCard.style.boxShadow = '0 0 16px rgba(251,146,60,.3)';
            } else {
                errCard.style.borderColor = 'rgba(251,146,60,.2)';
                errCard.style.boxShadow = '';
            }
        }

        const sys = data.system;

        // CPU
        animateValue('val-cpu-txt', sys.cpu_percent, '%');
        const barCpu = document.getElementById('bar-cpu');
        if (barCpu) {
            barCpu.style.width = `${sys.cpu_percent}%`;
            if (sys.cpu_percent > 85) barCpu.classList.add('bar-danger');
            else barCpu.classList.remove('bar-danger');
        }

        // GPU
        const gpuPct = sys.gpu_percent || 0;
        animateValue('val-gpu-txt', gpuPct, '%');
        const barGpu = document.getElementById('bar-gpu');
        if (barGpu) {
            barGpu.style.width = `${gpuPct}%`;
            if (gpuPct > 85) barGpu.classList.add('bar-danger');
            else barGpu.classList.remove('bar-danger');
        }

        // NPU
        const npuPct = sys.npu_percent || 0;
        animateValue('val-npu-txt', npuPct, '%');
        const barNpu = document.getElementById('bar-npu');
        if (barNpu) {
            barNpu.style.width = `${npuPct}%`;
            if (npuPct > 85) barNpu.classList.add('bar-danger');
            else barNpu.classList.remove('bar-danger');
        }

        // RAM
        animateValue('val-ram-gb', sys.ram_used_gb);
        animateValue('val-ram-txt', sys.ram_percent, '%');
        const barRam = document.getElementById('bar-ram');
        barRam.style.width = `${sys.ram_percent}%`;
        if (sys.ram_percent > 85) barRam.classList.add('bar-danger');
        else barRam.classList.remove('bar-danger');

        // Temp
        const temp = sys.temperature ? Math.round(sys.temperature) : 0;
        const tempEl = document.getElementById('val-temp-txt');
        tempEl.innerText = temp ? `${temp}°` : 'N/A';

        if (temp > 75) {
            tempEl.style.color = 'var(--red)';
            tempEl.style.borderColor = 'rgba(239,68,68,0.3)';
            tempEl.style.background = 'rgba(239,68,68,0.08)';
        } else {
            tempEl.style.color = '';
            tempEl.style.borderColor = '';
            tempEl.style.background = '';
        }

        if (data.detailed_detections) {
            renderDetailedDetectionCards(data.detailed_detections);
        } else {
            renderDetectionCards(data.detections);
        }

        // Fetch theft videos for pending count and thumbnail
        try {
            const theftRes = await fetch(`${API_BASE}/dashboard/detections?type=theft&limit=100`);
            const theftEvents = await theftRes.json();

            let pendingCount = 0;
            let latestVideoPath = null;

            theftEvents.forEach(ev => {
                const state = (ev.metadata && ev.metadata.review_state) ? ev.metadata.review_state : 'pending';
                if (state === 'pending') pendingCount++;
                if (!latestVideoPath && ev.metadata && ev.metadata.video_path) {
                    latestVideoPath = ev.metadata.video_path;
                }
            });

            animateValue('val-theft-pending', pendingCount);

            const thumbContainer = document.getElementById('theft-video-thumbnail');
            if (thumbContainer) {
                if (latestVideoPath) {
                    thumbContainer.innerHTML = `
                        <video style="width:100%; height:100%; object-fit:cover;" muted loop playsinline autoplay>
                            <source src="${latestVideoPath}" type="video/mp4">
                        </video>
                        <div style="position: absolute; background: rgba(0,0,0,0.5); padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; color: white;">Latest Detection</div>
                    `;
                    // Make sure container is relative for absolute positioning of text
                    thumbContainer.style.position = 'relative';
                } else {
                    thumbContainer.innerHTML = '<span style="font-size:0.8rem; color:var(--text-secondary);">No video available</span>';
                }
            }
        } catch (e) {
            console.error("Error fetching theft events for thumbnail:", e);
        }

        const d = new Date();
        document.getElementById('last-updated').innerText =
            `· ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;

    } catch (error) {
        console.error("Error fetching stats:", error);
    }
}

function renderDetailedDetectionCards(detailedDetections) {
    const container = document.getElementById('detection-cards');

    if (!detailedDetections || detailedDetections.length === 0) {
        container.innerHTML = '<div style="color:var(--text-secondary); grid-column:1/-1;">No model types found.</div>';
        return;
    }

    // Only clear if we currently have skeletons
    if (container.querySelector('.skeleton')) {
        container.innerHTML = '';
    }

    detailedDetections.forEach(det => {
        const type = det.type;
        const id = `det-card-${type.toLowerCase()}`;
        let card = document.getElementById(id);

        const isActiveOrDetected = det.cameras_running > 0 || det.active_now > 0 || det.total_today > 0;

        if (!isActiveOrDetected) {
            if (card) card.style.display = 'none';
            return;
        }
        if (card) card.style.display = '';

        let icon = '🤖';
        let color = '#3b82f6';
        if (type.includes('fire')) { icon = '🔥'; color = 'var(--red)'; }
        else if (type.includes('human') || type.includes('person') || type.includes('crowd')) { icon = '🚶'; color = 'var(--blue)'; }
        else if (type.includes('weapon') || type.includes('gun')) { icon = '🔫'; color = 'var(--purple)'; }
        else if (type.includes('motion') || type === 'mog2') { icon = '🟡'; color = 'var(--yellow)'; }
        else if (type.includes('theft')) { icon = '🦹'; color = '#a855f7'; }

        if (!card) {
            card = document.createElement('div');
            card.id = id;
            card.className = 'detect-card glass-panel';
            card.onclick = () => openDetectionList(type);

            card.innerHTML = `
                <div class="detect-card-icon">${icon}</div>
                <h3 style="margin-bottom:10px;">${det.display_name}</h3>
                <div style="display:flex; justify-content:space-between; width:100%; font-size:0.9rem;">
                    <div style="display:flex; flex-direction:column; align-items:center;">
                        <span style="color:var(--text-secondary); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px;">Active Cams</span>
                        <strong class="num-animate" id="det-active-${type.toLowerCase()}" style="color:${color}; font-size:1.4rem;">0</strong>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:center;">
                        <span style="color:var(--text-secondary); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px;">Detected (60s)</span>
                        <strong class="num-animate" id="det-detected-${type.toLowerCase()}" style="color:#fff; font-size:1.4rem;">0</strong>
                    </div>
                </div>
            `;
            container.appendChild(card);
        }

        animateValue(`det-active-${type.toLowerCase()}`, det.cameras_running);
        animateValue(`det-detected-${type.toLowerCase()}`, det.active_now);
    });
}

function renderDetectionCards(detections) {
    const container = document.getElementById('detection-cards');

    // Always ensure motion is tracked explicitly
    if (detections['Motion'] === undefined && detections['motion'] === undefined) {
        detections['Motion'] = 0;
    }

    if (Object.keys(detections).length === 0) {
        container.innerHTML = '<div style="color:var(--text-secondary); grid-column:1/-1;">No detections recorded today.</div>';
        return;
    }

    // Only clear if we currently have skeletons
    if (container.querySelector('.skeleton')) {
        container.innerHTML = '';
    }

    for (const [type, count] of Object.entries(detections)) {
        const id = `det-card-${type.toLowerCase()}`;
        let card = document.getElementById(id);

        let icon = '🟡';
        let color = '#3b82f6';
        if (type.toLowerCase() === 'fire') { icon = '🔥'; color = 'var(--red)'; }
        if (type.toLowerCase() === 'human') { icon = '🚶'; color = 'var(--blue)'; }
        if (type.toLowerCase() === 'weapon') { icon = '🔫'; color = 'var(--purple)'; }
        if (type.toLowerCase() === 'motion') { icon = '🟡'; color = 'var(--yellow)'; }

        if (!card) {
            card = document.createElement('div');
            card.id = id;
            card.className = 'detect-card glass-panel';
            card.onclick = () => openDetectionList(type);

            card.innerHTML = `
                <div class="detect-card-icon">${icon}</div>
                <h3>${type}</h3>
                <div class="detect-count num-animate" id="det-count-${type.toLowerCase()}" style="color: ${color}">0</div>
                <div class="detect-label">${type.toLowerCase() === 'motion' ? 'Cameras active now' : 'Total events today'}</div>
            `;
            container.appendChild(card);
        }

        animateValue(`det-count-${type.toLowerCase()}`, count);
    }
}

// ── Unified Detection Modal ─────────────────────────────────────────────
let currentDetectionType = null;

function closeDetectionModal() {
    document.getElementById('detection-modal').classList.remove('active');
    currentDetectionType = null;
}

async function openDetectionList(type) {
    currentDetectionType = type;
    const modal = document.getElementById('detection-modal');
    modal.classList.add('active');

    const isMotion = type.toLowerCase().includes('motion') || type === 'mog2';

    if (isMotion) {
        await loadMotionModal();
    } else {
        await loadDetectionModal(type);
    }
}

// Kept for legacy calls (top stat box click etc.)
async function openMotionModal() {
    await openDetectionList('motion');
}
function closeMotionModal() { closeDetectionModal(); }

async function loadMotionModal() {
    document.getElementById('detection-modal-title').innerHTML = '🟡 Active Motion Cameras';

    const thead = document.getElementById('detection-modal-thead');
    thead.innerHTML = `
        <tr>
            <th style="padding:10px; border-bottom:1px solid var(--glass-border);">Shop</th>
            <th style="padding:10px; border-bottom:1px solid var(--glass-border);">Camera</th>
            <th style="padding:10px; border-bottom:1px solid var(--glass-border);">Since</th>
        </tr>`;

    const tbody = document.getElementById('detection-modal-tbody');
    tbody.innerHTML = '<tr><td colspan="3" style="padding:14px; text-align:center;">Loading...</td></tr>';

    try {
        const allCameras = await (await fetch(`${API_BASE}/dashboard/motion-status`)).json();
        const active = allCameras.filter(c => c.active);

        if (active.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" style="padding:14px; text-align:center; color:var(--text-secondary);">No cameras currently detecting motion.</td></tr>';
            return;
        }

        tbody.innerHTML = active.map(cam => `
            <tr>
                <td style="padding:10px; border-bottom:1px solid var(--glass-border);"><strong>${cam.shop_name || '-'}</strong></td>
                <td style="padding:10px; border-bottom:1px solid var(--glass-border); color:var(--text-secondary);">${cam.camera_name || '-'}</td>
                <td style="padding:10px; border-bottom:1px solid var(--glass-border);">
                    <span style="color:var(--yellow); display:flex; align-items:center; gap:6px;">
                        <span class="live-indicator" style="width:6px;height:6px;box-shadow:none;"></span>
                        ${new Date(cam.started_at).toLocaleTimeString()}
                    </span>
                </td>
            </tr>`).join('');

    } catch (e) {
        document.getElementById('detection-modal-tbody').innerHTML =
            '<tr><td colspan="3" style="padding:14px; text-align:center; color:var(--red);">Error loading data.</td></tr>';
    }
}

async function loadDetectionModal(type) {
    const displayName = type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    document.getElementById('detection-modal-title').innerHTML = `📌 ${displayName} — Active Events`;

    const thead = document.getElementById('detection-modal-thead');
    thead.innerHTML = `
        <tr>
            <th style="padding:10px; border-bottom:1px solid var(--glass-border);">Shop</th>
            <th style="padding:10px; border-bottom:1px solid var(--glass-border);">Camera</th>
            <th style="padding:10px; border-bottom:1px solid var(--glass-border);">Since</th>
        </tr>`;

    const tbody = document.getElementById('detection-modal-tbody');
    tbody.innerHTML = '<tr><td colspan="3" style="padding:14px; text-align:center;">Loading...</td></tr>';

    const today = new Date().toISOString().split('T')[0];
    try {
        const events = await (await fetch(`${API_BASE}/dashboard/detections?type=${type}&date_str=${today}&limit=30&active_only=true`)).json();

        if (!events.length) {
            tbody.innerHTML = '<tr><td colspan="3" style="padding:14px; text-align:center; color:var(--text-secondary);">No cameras currently detecting this.</td></tr>';
            return;
        }

        tbody.innerHTML = events.map(ev => {
            const start = new Date(ev.detected_at).toLocaleTimeString();
            return `
                <tr>
                    <td style="padding:10px; border-bottom:1px solid var(--glass-border);"><strong>${ev.shop_name || '-'}</strong></td>
                    <td style="padding:10px; border-bottom:1px solid var(--glass-border); color:var(--text-secondary);">${ev.camera_name || '-'}</td>
                    <td style="padding:10px; border-bottom:1px solid var(--glass-border);">
                        <span style="color:var(--yellow); display:flex; align-items:center; gap:6px;">
                            <span class="live-indicator" style="width:6px;height:6px;box-shadow:none;"></span>
                            ${start}
                        </span>
                    </td>
                </tr>`;
        }).join('');

    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="3" style="padding:14px; text-align:center; color:var(--red);">Error loading data.</td></tr>';
    }
}

// ── Helpers ────────────────────────────────────────────────────────────
function animateValue(id, end, suffix = '') {
    const el = document.getElementById(id);
    if (!el) return;

    const start = oldValues[id] || 0;
    if (start === end) return;
    oldValues[id] = end;

    const duration = 600;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 4);
        const current = start + (end - start) * ease;

        el.innerText = Number.isInteger(end)
            ? Math.round(current) + suffix
            : current.toFixed(1) + suffix;

        if (progress < 1) requestAnimationFrame(update);
        else el.innerText = end + suffix;
    }
    requestAnimationFrame(update);
}

// ── Init & polling ──────────────────────────────────────────────────────
fetchStats();

setInterval(() => {
    fetchStats();
    // Live-refresh open modal
    const modal = document.getElementById('detection-modal');
    if (modal && modal.classList.contains('active') && currentDetectionType) {
        const isMotion = currentDetectionType.toLowerCase().includes('motion') || currentDetectionType === 'mog2';
        if (isMotion) loadMotionModal();
        else loadDetectionModal(currentDetectionType);
    }
}, 10000);

// ── Server Mode Toggle ──────────────────────────────────────────────────
async function initServerMode() {
    const select = document.getElementById('server-mode-select');
    if (!select) return;

    try {
        const res = await fetch(`${API_BASE}/dashboard/settings`);
        const data = await res.json();
        select.value = data.server_mode || 'standalone';
    } catch (e) {
        console.error("Failed to load server mode", e);
    }

    select.addEventListener('change', async (e) => {
        const newMode = e.target.value;
        const originalText = select.options[select.selectedIndex].text;
        select.options[select.selectedIndex].text = "Updating mode...";
        select.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/dashboard/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ server_mode: newMode })
            });
            const result = await res.json();
            
            if (res.ok && result.status === 'success') {
                // Short timeout to let the UI update and give backend threads time to stop/start
                setTimeout(() => {
                    alert(`Successfully switched to ${newMode === 'cloud' ? 'Cloud API' : 'Standalone'} mode!`);
                    fetchStats();
                }, 1000);
            } else {
                throw new Error(result.detail || "Failed to update");
            }
        } catch (err) {
            alert("Error updating server mode: " + err.message);
            console.error(err);
            // Revert on fail
            select.value = newMode === 'cloud' ? 'standalone' : 'cloud';
        } finally {
            select.options[select.selectedIndex].text = originalText;
            select.disabled = false;
        }
    });
}

initServerMode();