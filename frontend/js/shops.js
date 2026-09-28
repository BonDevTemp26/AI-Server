const API_BASE = '/api';
let modelsList = [];
let currentTimelineCameraId = null;
let allShopsData = []; // [{ shop, cameras }]
const expandedShops = new Set();
let cameraOnlineStatus = {}; // { camId: true/false }

// Track toggle state per camera: { cameraId: { motion: bool, modelId: bool, ... } }
const cameraState = {};

document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('timeline-date').valueAsDate = new Date();
    await fetchModels();
    await loadShops();
});

async function fetchModels() {
    try {
        const res = await fetch(`${API_BASE}/models`);
        modelsList = await res.json();
    } catch (e) {
        console.error("Failed to fetch models", e);
    }
}

// ─── Shop Loading ────────────────────────────────────────────────────────────

async function loadShops() {
    const container = document.getElementById('shop-list');
    container.innerHTML = shopListSkeleton();

    try {
        const res = await fetch(`${API_BASE}/shops`);
        const shops = await res.json();

        let serverCameraStates = {};
        try {
            const stateRes = await fetch(`${API_BASE}/workers/states/all`);
            if (stateRes.ok) serverCameraStates = await stateRes.json();
        } catch (err) { }

        allShopsData = [];
        for (let shop of shops) {
            const camRes = await fetch(`${API_BASE}/shops/${shop.id}/cameras`);
            const cameras = await camRes.json();
            allShopsData.push({ shop, cameras });

            cameras.forEach(cam => {
                cameraState[cam.id] = serverCameraStates[cam.id] || { motion: false };
            });
        }
        renderShopList();
        // Fetch live status badges after rendering
        refreshCameraStatus();
    } catch (e) {
        console.error("Failed to load shops", e);
        container.innerHTML = '<p style="color:var(--text-secondary); padding:20px;">Failed to load shops.</p>';
    }
}

async function refreshCameraStatus() {
    try {
        const res = await fetch(`${API_BASE}/dashboard/camera-status`);
        cameraOnlineStatus = await res.json();
        // Update all visible badges without re-rendering the whole list
        for (const [camId, stat] of Object.entries(cameraOnlineStatus)) {
            const badge = document.getElementById(`cam-badge-${camId}`);
            if (!badge) continue;

            const state = stat.state || (stat.online ? 'online' : 'offline');
            if (state === 'online') {
                badge.textContent = '● Online';
                badge.style.color = '#34d399';
                badge.title = '';
            } else if (state === 'error') {
                badge.textContent = '⚠ Error';
                badge.style.color = '#fb923c';
                badge.title = stat.error || 'Stream error';
            } else {
                badge.textContent = '● Offline';
                badge.style.color = '#f87171';
                badge.title = '';
            }
        }
    } catch (e) {
        console.error('Failed to refresh camera status', e);
    }
}

function shopListSkeleton() {
    return Array.from({ length: 3 }).map(() => `
        <div class="shop-item glass-panel">
            <div class="shop-header">
                <div class="skeleton-line" style="width:180px;"></div>
                <div class="skeleton-line" style="width:100px;"></div>
            </div>
        </div>
    `).join('');
}

// Renders allShopsData into #shop-list, applying the current search filter.
function renderShopList() {
    const container = document.getElementById('shop-list');
    const searchEl = document.getElementById('shop-search');
    const term = (searchEl ? searchEl.value : '').trim().toLowerCase();

    const filtered = !term ? allShopsData : allShopsData.filter(({ shop, cameras }) => {
        if (shop.name.toLowerCase().includes(term)) return true;
        return cameras.some(cam => cam.name.toLowerCase().includes(term) || (cam.rtsp_url || '').toLowerCase().includes(term));
    });

    if (!allShopsData.length) {
        container.innerHTML = '<p style="color:var(--text-secondary); padding:20px;">No shops yet — add your first shop above.</p>';
        return;
    }

    if (!filtered.length) {
        container.innerHTML = `<div class="no-results">🔎 No shops or cameras match "${escHtmlShops(term)}"</div>`;
        return;
    }

    container.innerHTML = filtered.map(({ shop, cameras }, i) => {
        const camerasHtml = cameras.length
            ? cameras.map(cam => buildCameraHtml(cam)).join('')
            : '<p style="color:var(--text-secondary); font-size:0.9rem; padding:10px 0;">No cameras added yet.</p>';

        const isExpanded = expandedShops.has(shop.id) || (!!term && cameras.length > 0);
        const animDelay = i * 0.05;

        // Build Shop-level model toggles (same layout as camera level)
        const byType = {};
        modelsList.forEach(m => {
            const t = (m.type || 'custom').toLowerCase();
            if (t === 'mog2' || t === 'motion') return;
            if (!byType[t]) byType[t] = [];
            byType[t].push(m);
        });

        let shopModelBtns = '';
        for (const [type, models] of Object.entries(byType)) {
            const icon = getModelIcon(type);
            if (models.length === 1) {
                const m = models[0];
                const anyActive = cameras.some(cam => cameraState[cam.id]?.[`model_${m.id}`]);
                shopModelBtns += `<button class="toggle-btn model-icon-btn ${anyActive ? 'active' : ''}" style="margin-right:2px; font-size:1.1rem; padding:4px 8px;" onclick="toggleShopModel(${shop.id}, ${m.id})" title="${m.name} (All Cameras)">${icon}</button>`;
            } else {
                const anyActive = models.some(m => cameras.some(cam => cameraState[cam.id]?.[`model_${m.id}`]));
                const safeType = type.replace(/[^a-z0-9]/gi, '');
                shopModelBtns += `
                    <div style="position:relative; display:inline-block; margin-right:2px;">
                        <button class="toggle-btn model-icon-btn ${anyActive ? 'active' : ''}" style="font-size:1.1rem; padding:4px 8px;" onclick="toggleShopModelDropdown(${shop.id},'${type}',this)" title="Select ${type} model (All Cameras)">
                            ${icon}<span style="font-size:0.6rem; opacity:.7;">▼</span>
                        </button>
                        <div id="shop-${shop.id}-dd-${safeType}" class="model-dropdown" style="display:none;"></div>
                    </div>`;
            }
        }

        return `
            <div class="shop-item glass-panel ${isExpanded ? 'expanded' : ''}" id="shop-${shop.id}" style="animation: fadeSlideUp 0.4s ease both; animation-delay: ${animDelay}s; border-color:var(--border-bright);">
                <div class="shop-header">
                    <div class="shop-title" onclick="toggleShopExpand(${shop.id})">
                        <span class="expand-icon" style="font-size:0.8rem; opacity:0.8; margin-right:4px;">${isExpanded ? '▼' : '▶'}</span> ${shop.name}
                    </div>
                    <div class="shop-actions">
                        <div style="display:flex; align-items:center; border-right:1px solid var(--glass-border); padding-right:10px; margin-right:5px;">
                            <button class="btn" style="padding: 4px 8px; font-size:0.9rem; margin-right:4px; background:rgba(245,158,11,.15); color:#fbbf24; border-color:rgba(245,158,11,.3);" onclick="toggleShopMotion(${shop.id})" title="Toggle Motion For All Cameras">🟡</button>
                            ${shopModelBtns}
                            <button class="btn" style="padding: 4px 8px; font-size:0.9rem; margin-left:4px; background:rgba(168,85,247,.15); color:#a855f7; border-color:rgba(168,85,247,.3); transition:all 0.2s;" onclick="toggleShopTheft(${shop.id})" title="Toggle Theft Detection For All Cameras" id="shop-theft-btn-${shop.id}">🦹</button>
                            <button class="btn" style="padding: 4px 8px; font-size:0.9rem; margin-left:4px; background:rgba(52,211,153,.15); color:#34d399; border-color:rgba(52,211,153,.3); transition:all 0.2s;" onclick="toggleShopAllAI(${shop.id})" title="Toggle ALL Models & Motion For All Cameras" id="shop-all-ai-btn-${shop.id}">⚡</button>
                        </div>
                        <button class="btn btn-primary" style="padding: 4px 10px;" onclick="openCameraModal(${shop.id})">+ Camera</button>
                        <button class="btn" style="padding: 4px 8px; background:rgba(255,255,255,.05);" onclick="openShopModal(${shop.id}, '${escapeQ(shop.name)}')">✏️</button>
                        <button class="btn" style="padding: 4px 8px; background:rgba(239,68,68,.15); border-color:rgba(239,68,68,.3); color:var(--red);" onclick="deleteShop(${shop.id})">🗑️</button>
                    </div>
                </div>
                <div class="camera-list" style="margin-top:16px; margin-left:14px; padding-left:18px; border-left:2px solid var(--glass-border);">
                    ${camerasHtml}
                </div>
            </div>
        `;
    }).join('');
}

function escHtmlShops(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ─── Model icon mapping ───────────────────────────────────────────────────────
const MODEL_TYPE_ICONS = {
    yolo_human: '🚶', human: '🚶',
    yolo_fire: '🔥', fire: '🔥',
    yolo_weapon: '🔫', weapon: '🔫',
    yolo_theft: '🕵️', theft: '🕵️',
    mog2: '🟡', motion: '🟡',
    custom: '🤖',
};

function getModelIcon(type) {
    if (!type) return '🤖';
    const key = type.toLowerCase();
    for (const [k, icon] of Object.entries(MODEL_TYPE_ICONS)) {
        if (key === k || key.includes(k)) return icon;
    }
    return '🤖';
}

function buildCameraHtml(cam) {
    const isMotionActive = cameraState[cam.id]?.motion ?? false;
    const isOnline = cameraOnlineStatus[cam.id]?.online;
    const badgeColor = isOnline === undefined ? '#94a3b8' : isOnline ? '#34d399' : '#f87171';
    const badgeText = isOnline === undefined ? '● Checking…' : isOnline ? '● Online' : '● Offline';

    const isTheftActive = cameraState[cam.id]?.theft ?? false;

    // Group models by type (excluding mog2/motion — those are the Motion button)
    const byType = {};
    modelsList.forEach(m => {
        const t = (m.type || 'custom').toLowerCase();
        if (t === 'mog2' || t === 'motion') return;
        if (!byType[t]) byType[t] = [];
        byType[t].push(m);
    });

    // Build icon buttons — one per type, skip if no models of that type
    let modelTypeBtns = '';
    for (const [type, models] of Object.entries(byType)) {
        const icon = getModelIcon(type);
        if (models.length === 1) {
            const m = models[0];
            const isActive = cameraState[cam.id]?.[`model_${m.id}`] ?? false;
            modelTypeBtns += `<button class="toggle-btn model-icon-btn ${isActive ? 'active' : ''}" id="cam-${cam.id}-model-${m.id}" onclick="toggleCameraModel(${cam.id}, ${m.id}, this)" title="${m.name}">${icon}</button>`;
        } else {
            const anyActive = models.some(m => cameraState[cam.id]?.[`model_${m.id}`]);
            const safeType = type.replace(/[^a-z0-9]/gi, '');
            modelTypeBtns += `
                <div style="position:relative; display:inline-block;">
                    <button class="toggle-btn model-icon-btn ${anyActive ? 'active' : ''}" onclick="toggleModelDropdown(${cam.id},'${type}',this)" title="Select ${type} model">
                        ${icon}<span style="font-size:0.6rem; opacity:.7;">▼</span>
                    </button>
                    <div id="cam-${cam.id}-dd-${safeType}" class="model-dropdown" style="display:none;"></div>
                </div>`;
        }
    }

    return `
        <div class="camera-item" id="camera-item-${cam.id}"
             style="display:flex; justify-content:space-between; align-items:center; padding:12px 16px; background:rgba(0,0,0,0.3); border:1px solid var(--border); border-radius:10px; margin-bottom:12px; transition:transform 0.2s, box-shadow 0.2s;"
             onmouseenter="this.style.transform='translateX(4px)'; this.style.borderColor='var(--border-bright)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)';"
             onmouseleave="this.style.transform='none'; this.style.borderColor='var(--border)'; this.style.boxShadow='none';">
            <div style="display:flex; flex-direction:column; gap:4px;">
                <span style="font-weight:600; font-size:0.95rem; display:flex; align-items:center; gap:8px;">
                    📹 ${cam.name}
                    <span id="cam-badge-${cam.id}" style="font-size:0.7rem; font-weight:600; color:${badgeColor};">${badgeText}</span>
                </span>
                <span style="font-size:0.75rem; color:var(--text-secondary); font-family:monospace;">${cam.rtsp_url}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; flex-wrap:nowrap;">
                <button class="motion-btn ${isMotionActive ? 'active' : ''}" id="cam-${cam.id}-motion"
                    onclick="toggleCameraMotion(${cam.id}, this)" title="Toggle Motion">🟡</button>
                ${modelTypeBtns}
                <button class="theft-btn ${isTheftActive ? 'active' : ''}" onclick="toggleCameraTheft(${cam.id})" title="Toggle Theft Detection" id="cam-theft-btn-${cam.id}">🦹</button>
                <button class="btn" style="padding:6px 10px; font-size:1rem; background:rgba(52,211,153,.15); border-color:rgba(52,211,153,.3); margin-left:4px;" onclick="toggleCameraAllAI(${cam.id})" title="Master AI Toggle" id="cam-all-ai-btn-${cam.id}">⚡</button>
                <button class="btn" style="padding:6px 10px; font-size:1rem; background:rgba(59,130,246,.15); border-color:rgba(59,130,246,.3);" onclick="openTimeline(${cam.id}, '${escapeQ(cam.name)}')" title="Timeline">📊</button>
                <button class="btn" style="padding:6px 10px; font-size:1rem; background:rgba(255,255,255,.05);" onclick="openCameraModal(${cam.shop_id}, ${cam.id}, '${escapeQ(cam.name)}', '${escapeQ(cam.rtsp_url)}')" title="Edit">✏️</button>
                <button class="btn" style="padding:6px 10px; font-size:1rem; background:rgba(239,68,68,.15); border-color:rgba(239,68,68,.3);" onclick="deleteCamera(${cam.id})" title="Delete">🗑️</button>
            </div>
        </div>
    `;
}

function toggleModelDropdown(camId, type, btn) {
    // Close other open dropdowns
    document.querySelectorAll('.model-dropdown').forEach(d => { if (d !== document.getElementById(`cam-${camId}-dd-${type.replace(/[^a-z0-9]/gi, '')}`)) d.style.display = 'none'; });
    const safeType = type.replace(/[^a-z0-9]/gi, '');
    const dd = document.getElementById(`cam-${camId}-dd-${safeType}`);
    if (!dd) return;
    if (dd.style.display === 'block') { dd.style.display = 'none'; return; }
    const models = modelsList.filter(m => (m.type || 'custom').toLowerCase() === type);
    const icon = getModelIcon(type);
    dd.innerHTML = models.map(m => {
        const isActive = cameraState[camId]?.[`model_${m.id}`] ?? false;
        return `<div class="model-dd-item ${isActive ? 'active' : ''}"
            onclick="toggleCameraModel(${camId}, ${m.id}, {classList:{toggle:function(c,v){},contains:function(){return ${false}},add:function(){},remove:function(){}}}); document.getElementById('cam-${camId}-dd-${safeType}').style.display='none';">
            ${icon} ${m.name}
        </div>`;
    }).join('');
    dd.style.display = 'block';
    setTimeout(() => {
        document.addEventListener('click', function handler(e) {
            if (!dd.contains(e.target) && e.target !== btn) { dd.style.display = 'none'; document.removeEventListener('click', handler); }
        });
    }, 0);
}

function escapeQ(s) {
    return String(s).replace(/'/g, "\\'");
}

// ─── Shop Expand ─────────────────────────────────────────────────────────────

function toggleShopExpand(shopId) {
    const el = document.getElementById(`shop-${shopId}`);
    el.classList.toggle('expanded');
    const isOpen = el.classList.contains('expanded');
    if (isOpen) expandedShops.add(shopId); else expandedShops.delete(shopId);
    const icon = el.querySelector('.expand-icon');
    icon.innerText = isOpen ? '▼' : '▶';
}

// ─── Model Toggles ───────────────────────────────────────────────────────────

async function toggleCameraMotion(camId, btn) {
    if (!cameraState[camId]) cameraState[camId] = {};
    const originalState = cameraState[camId].motion;
    cameraState[camId].motion = !cameraState[camId].motion;
    btn.classList.toggle('active', cameraState[camId].motion);

    const action = cameraState[camId].motion ? 'start' : 'stop';
    try {
        const mog2 = modelsList.find(m => m.type === 'mog2');
        let res;
        if (mog2) {
            res = await fetch(`${API_BASE}/workers/cameras/${camId}/models/${mog2.id}/${action}`, { method: 'POST' });
        } else {
            res = await fetch(`${API_BASE}/workers/cameras/${camId}/motion/${action}`, { method: 'POST' });
        }
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || "Failed to toggle motion");
        }
    } catch (e) {
        // Revert UI state on error
        cameraState[camId].motion = originalState;
        btn.classList.toggle('active', originalState);
        UI.toast(e.message || "Failed to toggle motion", "error");
    }
}

function toggleShopMotion(shopId) {
    const shopData = allShopsData.find(s => s.shop.id === shopId);
    if (!shopData || !shopData.cameras.length) return;

    // Check if currently ANY camera is active. If so, turn all OFF. Else, turn all ON.
    const anyActive = shopData.cameras.some(cam => cameraState[cam.id]?.motion);
    const targetState = !anyActive;

    shopData.cameras.forEach(cam => {
        const btn = document.getElementById(`cam-${cam.id}-motion`);
        if (btn) {
            // Force it to opposite of targetState so toggleCameraMotion does what we want
            cameraState[cam.id].motion = !targetState;
            toggleCameraMotion(cam.id, btn);
        }
    });
}

function toggleShopModel(shopId, modelId) {
    const shopData = allShopsData.find(s => s.shop.id === shopId);
    if (!shopData || !shopData.cameras.length) return;

    // If ANY camera has this model ON, we turn ALL OFF. Otherwise we turn ALL ON.
    const anyActive = shopData.cameras.some(cam => cameraState[cam.id]?.[`model_${modelId}`]);
    const targetState = !anyActive;

    shopData.cameras.forEach(cam => {
        const btn = document.getElementById(`cam-${cam.id}-model-${modelId}`);
        if (btn) {
            // Only trigger if current state doesn't match target
            const currentState = cameraState[cam.id]?.[`model_${modelId}`] ?? false;
            if (currentState !== targetState) {
                // The toggleCameraModel function handles state update & API call, 
                // but we need to fake a button class toggle since it takes a button element.
                toggleCameraModel(cam.id, modelId, btn);
            }
        } else {
            // Button might be hidden inside a dropdown, so we call toggleCameraModel with a fake btn object
            const currentState = cameraState[cam.id]?.[`model_${modelId}`] ?? false;
            if (currentState !== targetState) {
                toggleCameraModel(cam.id, modelId, { classList: { toggle: function (c) { return targetState; }, contains: function () { return !targetState }, add: function () { }, remove: function () { } } });
            }
        }
    });
}

function toggleShopModelDropdown(shopId, type, btn) {
    document.querySelectorAll('.model-dropdown').forEach(d => { if (d !== document.getElementById(`shop-${shopId}-dd-${type.replace(/[^a-z0-9]/gi, '')}`)) d.style.display = 'none'; });
    const safeType = type.replace(/[^a-z0-9]/gi, '');
    const dd = document.getElementById(`shop-${shopId}-dd-${safeType}`);
    if (!dd) return;
    if (dd.style.display === 'block') { dd.style.display = 'none'; return; }
    const models = modelsList.filter(m => (m.type || 'custom').toLowerCase() === type);
    const icon = getModelIcon(type);
    const shopData = allShopsData.find(s => s.shop.id === shopId);

    dd.innerHTML = models.map(m => {
        const anyActive = shopData ? shopData.cameras.some(cam => cameraState[cam.id]?.[`model_${m.id}`]) : false;
        return `<div class="model-dd-item ${anyActive ? 'active' : ''}"
            onclick="toggleShopModel(${shopId}, ${m.id}); document.getElementById('shop-${shopId}-dd-${safeType}').style.display='none';">
            ${icon} ${m.name} (All)
        </div>`;
    }).join('');

    dd.style.display = 'block';
    setTimeout(() => {
        document.addEventListener('click', function handler(e) {
            if (!dd.contains(e.target) && e.target !== btn) { dd.style.display = 'none'; document.removeEventListener('click', handler); }
        });
    }, 0);
}

function toggleCameraModel(camId, modelId, btn) {
    if (!cameraState[camId]) cameraState[camId] = { motion: false };

    const isActive = btn.classList.toggle('active');
    cameraState[camId][`model_${modelId}`] = isActive;

    const action = isActive ? 'start' : 'stop';
    fetch(`${API_BASE}/workers/cameras/${camId}/models/${modelId}/${action}`, { method: 'POST' }).catch(() => { });

    // If any detection model is active → auto-enable motion
    if (isActive) {
        const motionBtn = document.getElementById(`cam-${camId}-motion`);
        if (motionBtn && !motionBtn.classList.contains('active')) {
            toggleCameraMotion(camId, motionBtn);
        }
    }
}

async function toggleCameraTheft(camId) {
    if (!cameraState[camId]) cameraState[camId] = {};
    const btn = document.getElementById(`cam-theft-btn-${camId}`);

    let isActive = false;
    if (btn) {
        isActive = btn.classList.toggle('active');
        cameraState[camId].theft = isActive;
    } else {
        isActive = !cameraState[camId].theft;
        cameraState[camId].theft = isActive;
    }

    const action = isActive ? 'start' : 'stop';

    try {
        const res = await fetch(`${API_BASE}/workers/cameras/${camId}/theft/${action}`, { method: 'POST' });
        if (res.ok) {
            UI.toast(`Theft detection ${isActive ? 'started' : 'stopped'}!`, "success");
        } else {
            UI.toast("Failed to toggle theft detection", "error");
        }
    } catch (e) {
        UI.toast("Error toggling theft detection", "error");
    }
}

async function toggleShopTheft(shopId) {
    const shopData = allShopsData.find(s => s.shop.id === shopId);
    if (!shopData || !shopData.cameras.length) return;

    const anyActive = shopData.cameras.some(cam => cameraState[cam.id]?.theft);
    const targetState = !anyActive;

    const btn = document.getElementById(`shop-theft-btn-${shopId}`);
    if (btn) {
        if (targetState) {
            btn.style.background = '#a855f7';
            btn.style.color = '#fff';
        } else {
            btn.style.background = 'rgba(168,85,247,.15)';
            btn.style.color = '#a855f7';
        }
    }

    shopData.cameras.forEach(cam => {
        if (!cameraState[cam.id]) cameraState[cam.id] = {};
        cameraState[cam.id].theft = targetState;
        const camBtn = document.getElementById(`cam-theft-btn-${cam.id}`);
        if (camBtn) camBtn.classList.toggle('active', targetState);
    });

    const action = targetState ? 'start' : 'stop';
    try {
        const res = await fetch(`${API_BASE}/workers/shops/${shopId}/theft/${action}`, { method: 'POST' });
        if (res.ok) {
            UI.toast(`Theft detection ${targetState ? 'started' : 'stopped'} for shop!`, "success");
        } else {
            UI.toast("Failed to toggle shop theft detection", "error");
        }
    } catch (e) {
        UI.toast("Error toggling shop theft detection", "error");
    }
}

async function toggleCameraAllAI(camId) {
    if (!cameraState[camId]) cameraState[camId] = {};

    // Check if ANY AI/motion is active
    const isMotion = cameraState[camId].motion;
    const isTheft = cameraState[camId].theft;
    const isModel = modelsList.some(m => cameraState[camId][`model_${m.id}`]);

    // Target state is ON if everything is OFF, otherwise OFF
    const targetState = !(isMotion || isTheft || isModel);

    // Toggle Motion if needed
    if (!!isMotion !== targetState) {
        const motionBtn = document.getElementById(`cam-${camId}-motion`);
        if (motionBtn) await toggleCameraMotion(camId, motionBtn);
        // Note: toggleCameraMotion already handles the state and UI update internally.
        // Wait, if motionBtn is missing, we still need to set state. 
        if (!motionBtn) {
            cameraState[camId].motion = targetState;
            fetch(`${API_BASE}/workers/cameras/${camId}/motion/${targetState ? 'start' : 'stop'}`, { method: 'POST' }).catch(() => { });
        }
    }

    // Toggle Theft if needed
    if (!!isTheft !== targetState) {
        // Just call toggleCameraTheft directly, it handles toggling state natively
        // Wait, toggleCameraTheft acts as a pure toggle. We need to force a state.
        // Let's manually force it here to avoid race conditions.
        cameraState[camId].theft = targetState;
        const theftBtn = document.getElementById(`cam-theft-btn-${camId}`);
        if (theftBtn) theftBtn.classList.toggle('active', targetState);
        fetch(`${API_BASE}/workers/cameras/${camId}/theft/${targetState ? 'start' : 'stop'}`, { method: 'POST' }).catch(() => { });
    }

    // Toggle Models if needed
    for (const m of modelsList) {
        const t = (m.type || 'custom').toLowerCase();
        if (t === 'mog2' || t === 'motion') continue;

        const isActive = cameraState[camId][`model_${m.id}`];
        if (!!isActive !== targetState) {
            cameraState[camId][`model_${m.id}`] = targetState;
            fetch(`${API_BASE}/workers/cameras/${camId}/models/${m.id}/${targetState ? 'start' : 'stop'}`, { method: 'POST' }).catch(() => { });
        }
    }

    // Update the master button UI
    const masterBtn = document.getElementById(`cam-all-ai-btn-${camId}`);
    if (masterBtn) {
        if (targetState) {
            masterBtn.style.background = '#10b981';
            masterBtn.style.color = '#fff';
            masterBtn.style.boxShadow = '0 0 10px rgba(16,185,129,0.4)';
        } else {
            masterBtn.style.background = 'rgba(52,211,153,.15)';
            masterBtn.style.color = '#34d399';
            masterBtn.style.boxShadow = 'none';
        }
    }

    UI.toast(`All AI Models & Motion ${targetState ? 'Started' : 'Stopped'}!`, "success");

    // Re-render to update the model dropdowns and other buttons correctly
    renderShopList();
}

async function toggleShopAllAI(shopId) {
    const shopData = allShopsData.find(s => s.shop.id === shopId);
    if (!shopData || !shopData.cameras.length) return;

    // Check if ANY camera in shop has ANY AI/motion active
    let anyActive = false;
    for (const cam of shopData.cameras) {
        if (!cameraState[cam.id]) continue;
        if (cameraState[cam.id].motion) anyActive = true;
        if (cameraState[cam.id].theft) anyActive = true;
        if (modelsList.some(m => cameraState[cam.id][`model_${m.id}`])) anyActive = true;
    }

    const targetState = !anyActive;

    // Apply target state to all cameras
    for (const cam of shopData.cameras) {
        if (!cameraState[cam.id]) cameraState[cam.id] = {};

        // Motion
        if (!!cameraState[cam.id].motion !== targetState) {
            cameraState[cam.id].motion = targetState;
            fetch(`${API_BASE}/workers/cameras/${cam.id}/motion/${targetState ? 'start' : 'stop'}`, { method: 'POST' }).catch(() => { });
        }

        // Theft
        if (!!cameraState[cam.id].theft !== targetState) {
            cameraState[cam.id].theft = targetState;
            fetch(`${API_BASE}/workers/cameras/${cam.id}/theft/${targetState ? 'start' : 'stop'}`, { method: 'POST' }).catch(() => { });
        }

        // Models
        for (const m of modelsList) {
            const t = (m.type || 'custom').toLowerCase();
            if (t === 'mog2' || t === 'motion') continue;
            if (!!cameraState[cam.id][`model_${m.id}`] !== targetState) {
                cameraState[cam.id][`model_${m.id}`] = targetState;
                fetch(`${API_BASE}/workers/cameras/${cam.id}/models/${m.id}/${targetState ? 'start' : 'stop'}`, { method: 'POST' }).catch(() => { });
            }
        }
    }

    UI.toast(`All AI Models & Motion ${targetState ? 'Started' : 'Stopped'} for entire shop!`, "success");
    renderShopList();
}

// ─── Modals ───────────────────────────────────────────────────────────────────

function closeModals() {
    document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
}

function openShopModal(id = null, name = '') {
    document.getElementById('shop-id').value = id || '';
    document.getElementById('shop-name').value = name;
    document.getElementById('shop-modal-title').innerText = id ? 'Edit Shop' : 'Add Shop';
    document.getElementById('shop-modal').classList.add('active');
}

async function saveShop() {
    const id = document.getElementById('shop-id').value;
    const name = document.getElementById('shop-name').value.trim();
    if (!name) return UI.toast("Shop name required", "error");

    const method = id ? 'PUT' : 'POST';
    const url = id ? `${API_BASE}/shops/${id}` : `${API_BASE}/shops/`;

    try {
        const res = await fetch(url, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        if (res.ok) {
            closeModals();
            UI.toast(id ? "Shop updated" : "Shop created", "success");
            loadShops();
        } else {
            const d = await res.json();
            UI.toast("Error: " + d.detail, "error");
        }
    } catch (e) { console.error(e); UI.toast("Network error saving shop", "error"); }
}

async function deleteShop(id) {
    const ok = await UI.confirm("This will permanently delete the shop and all its cameras.", { title: "Delete this shop?" });
    if (!ok) return;
    try {
        await fetch(`${API_BASE}/shops/${id}`, { method: 'DELETE' });
        UI.toast("Shop deleted", "success");
        loadShops();
    } catch (e) { UI.toast("Failed to delete shop", "error"); }
}

function openCameraModal(shopId, cameraId = null, name = '', rtsp = '') {
    document.getElementById('camera-shop-id').value = shopId;
    document.getElementById('camera-id').value = cameraId || '';
    document.getElementById('camera-name').value = name;
    document.getElementById('camera-rtsp').value = rtsp;
    document.getElementById('camera-modal-title').innerText = cameraId ? 'Edit Camera' : 'Add Camera';
    document.getElementById('camera-modal').classList.add('active');
}

async function saveCamera() {
    const shopId = document.getElementById('camera-shop-id').value;
    const camId = document.getElementById('camera-id').value;
    const name = document.getElementById('camera-name').value.trim();
    const rtsp_url = document.getElementById('camera-rtsp').value.trim();

    if (!name || !rtsp_url) return UI.toast("Name and RTSP URL required", "error");

    try {
        let res;
        if (camId) {
            // PUT update existing camera
            res = await fetch(`${API_BASE}/cameras/${camId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, rtsp_url, is_active: true })
            });
        } else {
            res = await fetch(`${API_BASE}/shops/${shopId}/cameras`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, rtsp_url, is_active: true })
            });
        }
        if (res.ok) {
            closeModals();
            UI.toast(camId ? "Camera updated" : "Camera added", "success");
            loadShops();
        } else {
            const d = await res.json();
            UI.toast("Error: " + d.detail, "error");
        }
    } catch (e) { console.error(e); UI.toast("Network error saving camera", "error"); }
}

async function deleteCamera(id) {
    const ok = await UI.confirm("This camera and its detection history will be removed.", { title: "Delete this camera?" });
    if (!ok) return;
    try {
        await fetch(`${API_BASE}/cameras/${id}`, { method: 'DELETE' });
        UI.toast("Camera deleted", "success");
        loadShops();
    } catch (e) { UI.toast("Failed to delete camera", "error"); }
}

// ─── Bulk Insert ──────────────────────────────────────────────────────────────

function openBulkModal() {
    document.getElementById('bulk-result').innerHTML = '';
    document.getElementById('bulk-file').value = '';
    document.getElementById('bulk-file-name').textContent = '';
    document.getElementById('bulk-drop-zone').style.borderColor = '';
    document.getElementById('bulk-drop-zone').style.background = '';
    document.getElementById('bulk-upload-btn').disabled = false;
    document.getElementById('bulk-upload-btn').textContent = '⬆️  Upload & Process';
    document.getElementById('bulk-modal').classList.add('active');
}

function handleBulkDrop(e) {
    e.preventDefault();
    const dz = document.getElementById('bulk-drop-zone');
    dz.style.borderColor = '';
    dz.style.background = '';
    const file = e.dataTransfer.files[0];
    if (file) {
        // Assign to the hidden input via DataTransfer
        const dt = new DataTransfer();
        dt.items.add(file);
        document.getElementById('bulk-file').files = dt.files;
        showBulkFileName({ target: { files: [file] } });
    }
}

function showBulkFileName(e) {
    const file = e.target.files[0];
    if (!file) return;
    const dz = document.getElementById('bulk-drop-zone');
    dz.style.borderColor = 'var(--accent-green)';
    document.getElementById('bulk-file-name').textContent = `📁 ${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
}

async function uploadBulk() {
    const file = document.getElementById('bulk-file').files[0];
    if (!file) { UI.toast('Please select a CSV or Excel file first.', 'error'); return; }

    const btn = document.getElementById('bulk-upload-btn');
    btn.disabled = true;
    btn.innerHTML = '<span style="display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px;"></span> Processing…';
    document.getElementById('bulk-result').innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/bulk-insert/`, { method: 'POST', body: formData });
        const data = await res.json();

        if (res.ok && data.stats) {
            renderBulkResult(data.stats, null);
            UI.toast('Bulk insert complete', 'success');
            loadShops();
        } else {
            renderBulkResult(null, data.detail || 'Unknown error');
            UI.toast('Bulk insert failed', 'error');
        }
    } catch (e) {
        renderBulkResult(null, 'Network error: ' + e.message);
        UI.toast('Network error during bulk insert', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⬆️ &nbsp;Upload &amp; Process';
    }
}

function renderBulkResult(stats, errorMsg) {
    const el = document.getElementById('bulk-result');

    if (errorMsg) {
        el.innerHTML = `
            <div style="display:flex; align-items:center; gap:10px; margin-top:16px;
                        padding:12px 16px; border-radius:10px;
                        background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35);
                        color:#f87171; font-size:0.88rem;">
                <span style="font-size:1.2rem;">❌</span>
                <div><strong>Upload failed</strong><br><span style="opacity:.8;">${errorMsg}</span></div>
            </div>`;
        return;
    }

    const total = (stats.inserted || 0) + (stats.updated || 0) + (stats.skipped || 0) + (stats.errors || 0);

    el.innerHTML = `
        <div style="margin-top:18px;">
            <!-- Success banner -->
            <div style="display:flex; align-items:center; gap:10px; padding:10px 14px;
                        border-radius:9px; background:rgba(16,185,129,.1);
                        border:1px solid rgba(16,185,129,.3); color:#34d399;
                        font-size:0.88rem; font-weight:500; margin-bottom:14px;">
                <span style="font-size:1.2rem;">✅</span>
                Bulk insert complete &nbsp;·&nbsp; ${total} row${total !== 1 ? 's' : ''} processed
            </div>

            <!-- Stat Cards -->
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px;">
                ${statCard('Inserted', stats.inserted || 0, '✅', 'rgba(16,185,129,.12)', 'rgba(16,185,129,.35)', '#34d399')}
                ${statCard('Updated', stats.updated || 0, '🔄', 'rgba(59,130,246,.12)', 'rgba(59,130,246,.35)', '#60a5fa')}
                ${statCard('Skipped', stats.skipped || 0, '⏭️', 'rgba(245,158,11,.12)', 'rgba(245,158,11,.35)', '#fbbf24')}
                ${statCard('Errors', stats.errors || 0, '❌', 'rgba(239,68,68,.12)', 'rgba(239,68,68,.35)', '#f87171')}
            </div>
        </div>`;
}

function statCard(label, value, icon, bg, border, color) {
    return `
        <div style="background:${bg}; border:1px solid ${border}; border-radius:10px;
                    padding:14px 8px; text-align:center; color:${color};">
            <div style="font-size:1.1rem; margin-bottom:4px;">${icon}</div>
            <div style="font-size:1.9rem; font-weight:700; line-height:1;">${value}</div>
            <div style="font-size:0.7rem; margin-top:5px; opacity:0.75;
                        text-transform:uppercase; letter-spacing:0.06em;">${label}</div>
        </div>`;
}

// ─── Timeline ────────────────────────────────────────────────────────────────

function openTimeline(camId, camName) {
    currentTimelineCameraId = camId;
    document.getElementById('timeline-camera-name').innerText = camName;
    document.getElementById('timeline-modal').classList.add('active');
    loadTimeline();
}

async function loadTimeline() {
    if (!currentTimelineCameraId) return;
    const dateStr = document.getElementById('timeline-date').value;

    // Clear both bars
    document.getElementById('timeline-bar-motion').innerHTML = '';
    document.getElementById('timeline-bar-detect').innerHTML = '';

    try {
        const res = await fetch(`${API_BASE}/shops/cameras/${currentTimelineCameraId}/timeline?date=${dateStr}`);
        const data = await res.json();
        renderTimeline(data.detection_periods, dateStr);
    } catch (e) {
        console.error("Failed to load timeline", e);
    }
}

function renderTimeline(detections, dateStr) {
    const tooltip = document.getElementById('timeline-tooltip');
    // Parse start-of-day timestamp in local time
    const startOfDay = new Date(dateStr + 'T00:00:00').getTime();
    const msInDay = 24 * 60 * 60 * 1000;

    detections.forEach(det => {
        const dStart = new Date(det.start).getTime();
        const dEnd = new Date(det.end).getTime();

        let startPct = Math.max(0, ((dStart - startOfDay) / msInDay) * 100);
        let endPct = Math.min(100, ((dEnd - startOfDay) / msInDay) * 100);
        const width = Math.max(0.3, endPct - startPct);

        const type = (det.type || '').toLowerCase();

        // Determine which row and color
        let barId, color;
        if (type === 'motion' || type === 'mog2') {
            barId = 'timeline-bar-motion';
            color = '#f59e0b'; // amber
        } else {
            barId = 'timeline-bar-detect';
            if (type === 'human') color = '#3b82f6'; // blue
            else if (type === 'fire') color = '#ef4444'; // red
            else if (type === 'weapon') color = '#a855f7'; // purple
            else color = '#10b981'; // green (other)
        }

        const seg = document.createElement('div');
        seg.className = 'timeline-segment';
        seg.style.left = `${startPct}%`;
        seg.style.width = `${width}%`;
        seg.style.background = color;

        // Tooltip
        seg.addEventListener('mouseenter', (e) => {
            const tStart = new Date(det.start).toLocaleTimeString();
            const tEnd = new Date(det.end).toLocaleTimeString();
            const cam = det.camera_name || '';
            tooltip.innerHTML = `<strong>${type.toUpperCase()}</strong>${cam ? ' · ' + cam : ''}<br>${tStart} → ${tEnd}`;
            tooltip.style.display = 'block';
        });
        seg.addEventListener('mousemove', (e) => {
            tooltip.style.left = (e.clientX + 14) + 'px';
            tooltip.style.top = (e.clientY + 14) + 'px';
        });
        seg.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });

        document.getElementById(barId).appendChild(seg);
    });
}

// Auto-refresh camera status badges every 30 seconds
setInterval(refreshCameraStatus, 30000);
