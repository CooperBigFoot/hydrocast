import importlib
import sys
from pathlib import Path


def pytest_configure(config):  # noqa: ARG001
    """Early hook: fix sys.path so installed packages win over workspace dirs.

    With --import-mode=importlib, pytest resolves coach/ (the submodule
    directory) as a namespace package, shadowing the real installed coach
    package whose code lives at coach/src/coach/. We fix this by:
      1. Removing the workspace root from sys.path
      2. Pre-importing coach from the correct src location so it's cached
         in sys.modules before collection tries to resolve it.
    """
    root = str(Path(__file__).parent)

    # Remove workspace root so bare directory names don't shadow packages
    while root in sys.path:
        sys.path.remove(root)

    # Ensure each submodule's src/ directory is on sys.path
    for pkg in ("coach", "data-leak", "time-flies"):
        src = str(Path(__file__).parent / pkg / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

    # If 'coach' was already (mis-)imported as a namespace package from the
    # bare directory, purge it and all submodules so the real package wins.
    stale = [k for k in sys.modules if k == "coach" or k.startswith("coach.")]
    for k in stale:
        del sys.modules[k]

    # Force-import coach from the now-correct path so it's cached
    import coach  # noqa: F811

    importlib.reload(coach)
