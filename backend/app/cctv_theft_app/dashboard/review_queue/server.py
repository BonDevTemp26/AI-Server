"""Deprecated entrypoint — the review queue moved into the unified dashboard.

The review page now lives at http://localhost:8080/review inside
``dashboard/server.py`` (navbar: live streaming, detection & tracking,
temporal understanding, VLM verification, alert review). The API endpoints
(``/api/reviews``, ``/api/reviews/{id}/label``, ``/clips/{name}``) are
unchanged, so existing tooling keeps working.

    python -m dashboard.server                    # preferred
    python -m dashboard.review_queue.server       # this shim, same app
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.server import app  # noqa: E402,F401  (re-export for uvicorn)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
