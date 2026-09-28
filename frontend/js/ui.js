/* ═══════════════════════════════════════════════════════════════
   Shared UI toolkit — toasts, confirm modal, mobile sidebar
   Loaded on every page before the page-specific script.
   ═══════════════════════════════════════════════════════════════ */

const UI = (() => {

    // ── Toasts ────────────────────────────────────────────────────────────
    let toastContainer = null;

    function ensureToastContainer() {
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
        return toastContainer;
    }

    const ICONS = { success: '✅', error: '⚠️', info: 'ℹ️' };

    function toast(message, type = 'info', duration = 4200) {
        const container = ensureToastContainer();
        const el = document.createElement('div');
        el.className = `toast ${type}`;
        el.innerHTML = `
            <span class="toast-icon">${ICONS[type] || ICONS.info}</span>
            <span class="toast-msg"></span>
        `;
        el.querySelector('.toast-msg').textContent = message;
        container.appendChild(el);

        const remove = () => {
            el.classList.add('leaving');
            setTimeout(() => el.remove(), 220);
        };
        el.addEventListener('click', remove);
        setTimeout(remove, duration);
        return el;
    }

    // ── Confirm modal (Promise-based, replaces window.confirm) ─────────────
    function confirm(message, { title = 'Are you sure?', danger = true, confirmLabel = 'Delete', cancelLabel = 'Cancel' } = {}) {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'confirm-overlay';
            overlay.innerHTML = `
                <div class="confirm-box">
                    <div class="confirm-icon">${danger ? '⚠️' : '❓'}</div>
                    <h3></h3>
                    <p></p>
                    <div class="confirm-actions">
                        <button class="confirm-cancel"></button>
                        <button class="${danger ? 'confirm-danger' : ''}"></button>
                    </div>
                </div>
            `;
            overlay.querySelector('h3').textContent = title;
            overlay.querySelector('p').textContent = message;
            const [cancelBtn, okBtn] = overlay.querySelectorAll('.confirm-actions button');
            cancelBtn.textContent = cancelLabel;
            okBtn.textContent = confirmLabel;

            const close = (result) => {
                overlay.style.animation = 'fadeIn .15s ease reverse';
                setTimeout(() => overlay.remove(), 140);
                resolve(result);
            };

            cancelBtn.onclick = () => close(false);
            okBtn.onclick = () => close(true);
            overlay.addEventListener('click', (e) => { if (e.target === overlay) close(false); });
            document.addEventListener('keydown', function esc(e) {
                if (e.key === 'Escape') { close(false); document.removeEventListener('keydown', esc); }
            });

            document.body.appendChild(overlay);
            okBtn.focus();
        });
    }

    // ── Mobile sidebar toggle ────────────────────────────────────────────
    function initMobileNav() {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        let backdrop = document.querySelector('.sidebar-backdrop');
        if (!backdrop) {
            backdrop = document.createElement('div');
            backdrop.className = 'sidebar-backdrop';
            document.body.appendChild(backdrop);
        }

        // Inject hamburger button into the top header if not present
        const header = document.querySelector('.top-header');
        if (header && !header.querySelector('.hamburger-btn')) {
            const btn = document.createElement('button');
            btn.className = 'hamburger-btn';
            btn.innerHTML = '☰';
            btn.setAttribute('aria-label', 'Toggle menu');
            header.prepend(btn);
            btn.addEventListener('click', () => {
                sidebar.classList.toggle('open');
                backdrop.classList.toggle('open');
            });
        }

        backdrop.addEventListener('click', () => {
            sidebar.classList.remove('open');
            backdrop.classList.remove('open');
        });

        // Close on nav link tap (mobile)
        sidebar.querySelectorAll('.nav-item').forEach(a => {
            a.addEventListener('click', () => {
                sidebar.classList.remove('open');
                backdrop.classList.remove('open');
            });
        });
    }

    document.addEventListener('DOMContentLoaded', initMobileNav);

    return { toast, confirm };
})();

// ── Relative time helper (used across pages) ───────────────────────────────
function timeAgo(isoOrDate) {
    const d = (isoOrDate instanceof Date) ? isoOrDate : new Date(isoOrDate);
    const diffSec = Math.floor((Date.now() - d.getTime()) / 1000);
    if (diffSec < 5) return 'just now';
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
}
