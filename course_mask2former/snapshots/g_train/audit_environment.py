"""Validate the dependency snapshot and command paths; record known headless aliases."""
from importlib import metadata
import json
from pathlib import Path
import subprocess

from packaging.requirements import Requirement


def main():
    root = Path('/workspace')
    checked = {}
    for line in (root / 'scripts/constraints-resolved.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        requirement = Requirement(line)
        version = metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f'{requirement}: installed {version}')
        checked[requirement.name] = version
    # These entry points use shebangs from the original Conda installation prefix.
    pip = subprocess.run(['pip', '--version'], check=True, text=True, capture_output=True)
    subprocess.run(['torchrun', '--help'], check=True, text=True, capture_output=True)
    check = subprocess.run(['python', '-m', 'pip', 'check'], text=True, capture_output=True)
    issues = check.stdout.strip().splitlines() if check.returncode else []
    expected = {
        'mmcv 2.1.0 requires opencv-python, which is not installed.',
        'mmengine 0.10.7 requires opencv-python, which is not installed.',
    }
    if check.returncode not in (0, 1) or any(issue not in expected for issue in issues):
        raise RuntimeError(check.stdout + check.stderr)
    if any(expected_line not in issues for expected_line in expected):
        # No missing alias warnings is also fine; other installed packages may supply it.
        print('Dependency metadata changed; inspect pip check output if unexpected.')
    report = dict(
        status='passed', installed_versions=checked, pip_entry_point=pip.stdout.strip(),
        torchrun_entry_point='passed', dependency_metadata_warnings=issues,
        warning_explanation='OpenCV-headless provides cv2; full OpenCV intentionally absent.',
    )
    destination = root / 'outputs/smoke/environment_audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
