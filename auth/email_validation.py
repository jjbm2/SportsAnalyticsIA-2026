from __future__ import annotations

import re


EMAIL_PATTERN = re.compile(
    r"^[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$",
    re.IGNORECASE,
)

BLOCKED_LOCAL_PARTS = {
    "asdf",
    "demo",
    "example",
    "fake",
    "noemail",
    "prueba",
    "qwerty",
    "temp",
    "temporary",
    "test",
    "testing",
}

BLOCKED_DOMAINS = {
    "10minutemail.com",
    "dispostable.com",
    "example.invalid",
    "fake.com",
    "guerrillamail.com",
    "mailinator.com",
    "temp-mail.org",
    "tempmail.com",
    "test.com",
    "yopmail.com",
}


def normalize_email(email: str) -> str:
    normalized = str(email or "").strip().casefold()
    if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
        raise ValueError("Correo electrónico no válido")
    local_part, domain = normalized.rsplit("@", 1)
    if (
        len(local_part) > 64
        or local_part.startswith(".")
        or local_part.endswith(".")
        or ".." in local_part
        or len(domain) > 253
    ):
        raise ValueError("Correo electrónico no válido")
    return normalized


def validate_registration_email(email: str) -> str:
    """Reject obvious placeholder/disposable addresses for new accounts."""
    normalized = normalize_email(email)
    local_part, domain = normalized.rsplit("@", 1)
    base_local = local_part.split("+", 1)[0]
    if base_local in BLOCKED_LOCAL_PARTS or domain in BLOCKED_DOMAINS:
        raise ValueError(
            "Usa un correo electrónico real; no se permiten correos de prueba o temporales"
        )
    return normalized
