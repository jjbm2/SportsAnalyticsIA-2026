from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# A 429 is intentionally not retried here. Retrying it inside urllib3 can
# create another burst; API-Sports traffic is paced centrally instead.
TRANSIENT_STATUS_CODES = (500, 502, 503, 504)


def build_retry_session() -> requests.Session:
    """Create a bounded HTTP client for transient provider failures."""
    retries = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.4,
        status_forcelist=TRANSIENT_STATUS_CODES,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
