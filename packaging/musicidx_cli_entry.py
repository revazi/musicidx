"""PyInstaller entrypoint for the bundled MusicIdx CLI sidecar."""

from __future__ import annotations

from musicidx.cli import app

if __name__ == "__main__":
    app()
