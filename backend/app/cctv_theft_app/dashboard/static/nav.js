// Shared navbar — injected into <nav class="topbar"> on every page.
(function () {
  const SECTIONS = [
    ["/", "Live Streaming"],
    ["/detection", "Per-Frame Detection & Tracking"],
    ["/temporal", "Temporal Understanding"],
    ["/vlm", "VLM Verification"],
    ["/review", "Alert Pipeline · Human Review"],
  ];
  const here = location.pathname.replace(/\/$/, "") || "/";
  const nav = document.querySelector("nav.topbar");
  
  if (nav) {
    nav.innerHTML =
      '<a class="brand" href="/">🎥 CCTV Theft Detection</a>' +
      SECTIONS.map(([path, label]) =>
        `<a class="navlink ${path === here ? "active" : ""}" href="${path}">${label}</a>`
      ).join("") +
      `<button id="feed-toggle" onclick="toggleActivityFeed()" class="ghost">
         Activity <span class="badge" id="feed-badge" style="display:none;">0</span>
       </button>`;
  }

  // Wrap body contents in layout-wrapper for CSS layout
  const wrapper = document.createElement('div');
  wrapper.className = 'layout-wrapper';
  const mainEl = document.querySelector('main');
  if (nav && mainEl) {
    nav.parentNode.insertBefore(wrapper, nav);
    wrapper.appendChild(nav);
    wrapper.appendChild(mainEl);
  }

  // Inject Activity Feed Sidebar
  const sidebar = document.createElement('aside');
  sidebar.id = 'activity-sidebar';
  sidebar.innerHTML = `
    <div class="activity-header">
      <h3>Live Activity Feed</h3>
      <button class="close-btn" onclick="toggleActivityFeed()">×</button>
    </div>
    <div class="activity-feed-content" id="activity-feed-content">
      <div style="color:var(--muted); text-align:center; padding:2rem 0; font-size:0.9em;">
        Loading activities...
      </div>
    </div>
  `;
  document.body.appendChild(sidebar);
})();

let feedOpen = false;
window._unreadActivities = 0;
let lastReviewCount = -1;

function toggleActivityFeed() {
  const sidebar = document.getElementById('activity-sidebar');
  const wrapper = document.querySelector('.layout-wrapper');
  feedOpen = !feedOpen;
  if (feedOpen) {
    sidebar.classList.add('open');
    if (wrapper) wrapper.classList.add('sidebar-open');
    const badge = document.getElementById('feed-badge');
    if (badge) badge.style.display = 'none';
    window._unreadActivities = 0;
  } else {
    sidebar.classList.remove('open');
    if (wrapper) wrapper.classList.remove('sidebar-open');
  }
}

async function fetchActivities() {
  try {
    const reviews = await apiGet("/api/reviews");
    // /api/reviews already returns sorted by created_at desc, but let's be sure
    reviews.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    
    const content = document.getElementById('activity-feed-content');
    if (!content) return;
    
    if (lastReviewCount !== -1 && reviews.length > lastReviewCount && !feedOpen) {
      window._unreadActivities += (reviews.length - lastReviewCount);
      const badge = document.getElementById('feed-badge');
      if (badge) {
        badge.textContent = window._unreadActivities;
        badge.style.display = 'inline-block';
      }
    }
    lastReviewCount = reviews.length;
    
    if (reviews.length === 0) {
      content.innerHTML = '<div style="color:var(--muted); text-align:center; padding:2rem 0; font-size:0.9em;">No activities found.</div>';
      return;
    }
    
    content.innerHTML = reviews.map(r => {
      const date = new Date((r.created_at || 0) * 1000);
      const timeStr = date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
      const labelTag = r.human_label === 'confirmed_theft' 
          ? '<span class="tag bad activity-tag">Theft Confirmed</span>' 
          : (r.human_label === 'false_positive' ? '<span class="tag ok activity-tag">False Positive</span>' : '<span class="tag warn activity-tag">Pending Review</span>');
          
      return `
        <div class="activity-item">
          <div class="time">${timeStr}</div>
          <div class="title">Event: ${esc(r.event_id || 'Unknown')}</div>
          <div class="desc">A potential theft was detected in the stream.</div>
          ${labelTag}
        </div>
      `;
    }).join("");
    
  } catch (e) {
    console.error("Activity feed poll error:", e);
  }
}

// Start polling
setInterval(fetchActivities, 5000);
setTimeout(fetchActivities, 500);

// Small shared helpers used by the pages.
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function apiSend(url, method, body) {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}
function showMsg(el, text, cls) {
  el.className = "msg " + cls;
  el.textContent = text;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
