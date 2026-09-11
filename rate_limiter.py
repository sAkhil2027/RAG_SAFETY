from fastapi import Request, HTTPException
import time
from collections import defaultdict

# In‑memory per‑IP request counter: {ip: (count, window_start)}
_rate_limits = defaultdict(lambda: (0, time.time()))
_MAX_REQUESTS_PER_MINUTE = 60

async def rate_limiter(request: Request):
    """Simple rate‑limiter middleware.
    Allows up to ``_MAX_REQUESTS_PER_MINUTE`` requests per client IP per minute.
    Raises ``HTTPException`` 429 when the limit is exceeded.
    """
    client_ip = request.client.host if request.client else "unknown"
    count, start = _rate_limits[client_ip]
    now = time.time()
    # Reset the window after a minute
    if now - start > 60:
        _rate_limits[client_ip] = (1, now)
    else:
        if count >= _MAX_REQUESTS_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too Many Requests")
        _rate_limits[client_ip] = (count + 1, start)
    # No return value needed – just let processing continue
