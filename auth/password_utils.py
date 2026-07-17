from __future__ import annotations

import bcrypt


MIN_PASSWORD_LENGTH = 8
MAX_BCRYPT_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    validate_password(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def validate_password(password: str) -> None:
    if not isinstance(password, str):
        raise ValueError("La contraseña no es válida")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError("La contraseña debe tener al menos 8 caracteres")
    if len(password.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
        raise ValueError("La contraseña es demasiado larga")
    if password != password.strip():
        raise ValueError("La contraseña no debe iniciar ni terminar con espacios")
    if not any(character.isalpha() for character in password):
        raise ValueError("La contraseña debe incluir al menos una letra")
    if not any(character.isdigit() for character in password):
        raise ValueError("La contraseña debe incluir al menos un número")
