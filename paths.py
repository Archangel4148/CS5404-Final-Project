import platform
from pathlib import Path

def fix_path(p: str | Path) -> Path:
    """
    Converts Windows paths (C:\\...) into WSL paths (/mnt/c/...)
    ONLY when running under WSL.
    Otherwise, returns a normal OS path.
    """
    p = str(p)

    # Detect WSL
    is_wsl = "microsoft" in platform.uname().release.lower()

    # If not WSL → return normal Path()
    if not is_wsl:
        return Path(p)

    # If WSL but the path is already Linux
    if p.startswith("/"):
        return Path(p)

    # If WSL and path is Windows style (C:\...)
    if ":" in p:
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")

    return Path(p)
