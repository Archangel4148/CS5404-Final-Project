import platform
from pathlib import Path

def fix_path(p: str | Path) -> Path:
    """Converts paths to the correct format (WSL vs Windows)."""
    p = str(p)

    # Detect WSL
    is_wsl = "microsoft" in platform.uname().release.lower()

    # If not WSL, return normal Path()
    if not is_wsl:
        return Path(p)

    # If WSL, handle it:
    if p.startswith("/"):  # Path is already Linux-style, leave it alone
        return Path(p)
    if ":" in p:  # Path is Windows-style, convert it
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")

    return Path(p)
