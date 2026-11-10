"""Build the wheel and check what an installer would see. Slow: needs ``uv`` and a build."""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_wheel_contains_package_and_console_script(tmp_path: Path):
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("findex-*.whl")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        entry_points = next(z.read(n).decode() for n in names if n.endswith("entry_points.txt"))
        metadata = next(z.read(n).decode() for n in names if n.endswith("/METADATA"))
    assert "findex/cli.py" in names and "findex/py.typed" in names
    assert not any(n.startswith("tests/") for n in names)  # src layout: tests are not shipped
    assert "[console_scripts]" in entry_points and "findex = findex.cli:app" in entry_points
    assert "Requires-Dist: typer" in metadata and "Requires-Dist: rich" in metadata
    assert "Version: 0.4.0" in metadata
