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
    for pkg in ("coach", "data-leak", "time-flies", "launchpad"):
        src = str(Path(__file__).parent / pkg / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

    # If packages were already (mis-)imported as namespace packages from
    # bare directories, purge them and all submodules so the real packages win.
    for pkg_name in ("coach", "launchpad"):
        stale = [k for k in sys.modules if k == pkg_name or k.startswith(f"{pkg_name}.")]
        for k in stale:
            del sys.modules[k]

    # Force-import packages from the now-correct paths so they're cached
    import coach  # noqa: F811

    importlib.reload(coach)

    import launchpad  # noqa: F811

    importlib.reload(launchpad)
