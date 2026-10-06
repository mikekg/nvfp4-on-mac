"""Offline regression check: python3 tests/test_makefile.py."""

import re
import subprocess
import tempfile
from pathlib import Path


def run():
    repo = Path(__file__).resolve().parents[1]
    install = subprocess.check_output(
        ["make", "-n", "install"], cwd=repo, text=True,
    )
    assert 'install -q "mlx>=0.32.3"' in install
    assert "mlx.git@main" not in install
    assert "install -q -e mlx-lm\n" in install
    patches = sorted((repo / "patches").glob("*.patch"))
    with tempfile.TemporaryDirectory() as directory:
        checkout = Path(directory) / "mlx-lm"
        source = checkout / "mlx_lm/generate.py"
        source.parent.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        lines = ["\n"] * 500
        for patch in patches:
            position = None
            for line in patch.read_text().splitlines(keepends=True):
                if line.startswith("@@"):
                    position = int(re.match(r"@@ -(\d+)", line)[1]) - 1
                elif position is not None and line.startswith((" ", "-")):
                    lines[position] = line[1:]
                    position += 1
        source.write_text("".join(lines))

        def apply():
            return subprocess.run(
                ["make", "patch", f"MLX_LM_DIR={checkout}"],
                cwd=repo, capture_output=True, text=True,
            )

        first = apply()
        assert first.returncode == 0, first.stderr
        for patch in patches:
            subprocess.run(
                ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch)],
                check=True,
            )
        second = apply()
        assert second.returncode == 0, second.stderr
        assert second.stdout.count("already applied") == len(patches)
        source.write_text("incompatible upstream code\n")
        failed = apply()
        assert failed.returncode != 0
        assert "already applied" not in failed.stdout

    print("Makefile patch regression: PASS")


if __name__ == "__main__":
    run()
