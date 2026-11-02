"""Shared account fields for the local CLI transports.

Claude Code, Cursor, and Codex each learn who is signed in from their own
executable. They all report that identity through this one status shape, and
the Providers page renders it through one line. An email is stored masked.
"""

from __future__ import annotations


def mask_account(email: str) -> str:
    if "@" not in email:
        return ""
    local, domain = email.split("@", 1)
    if not local or not domain:
        return ""
    return f"{local[0]}…@{domain}"


def link_status(
    *,
    binary: str,
    transport: str,
    installed: bool = False,
    version: str = "",
    logged_in: bool = False,
    auth_method: str = "",
    subscription: str = "",
    account: str = "",
    error: str = "",
    login_started: bool = False,
    login_url: str = "",
) -> dict:
    return {
        "binary": binary,
        "transport": transport,
        "installed": installed,
        "version": version,
        "logged_in": logged_in,
        "auth_method": auth_method,
        "subscription": subscription,
        "account": account,
        "error": error,
        "login_started": login_started,
        "login_url": login_url,
    }
