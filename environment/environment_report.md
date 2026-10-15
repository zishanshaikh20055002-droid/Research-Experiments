# Isolated Baseline Environment

## Result

**SUCCESS.** A new, isolated virtual environment was created without modifying
the bundled or global Python installation.

## Base Interpreter Inspection

- Base executable: `<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- Base Python: `3.12.14`, `64bit (AMD64)`
- Base pip: `25.0.1` in the new environment; source interpreter pip was `26.2.1`.
- Base interpreter supports `venv`.
- Before setup: NumPy and pandas were present in the bundled runtime; SciPy and scikit-learn were absent.
- No project-local virtual environment or primary dependency manifest was found. An archived shuffle-controls cloud package has its own requirements file and was not reused.

## Environment Created

- Environment path: `D:\venvs\xjtu-baselines`
- Environment Python: `D:\venvs\xjtu-baselines\Scripts\python.exe`
- Creation command:

```powershell
& '<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv 'D:\venvs\xjtu-baselines'
```

## Installation

Only the new virtual environment was modified. No global packages were installed,
upgraded, or removed.

```powershell
& 'D:\venvs\xjtu-baselines\Scripts\python.exe' -m pip install --disable-pip-version-check scikit-learn pandas
```

- Installation log: `scikit_learn_installation.log`
- Exact resolved packages: `pip_freeze.txt`

## Verified Versions

- Python: `3.12.14` (64-bit)
- NumPy: `2.5.4`
- SciPy: `1.18.1`
- pandas: `3.0.6`
- scikit-learn: `1.9.1`

## Smoke Test

Using only a six-row synthetic array, import, fit, and predict succeeded for:

- `Ridge`
- `ExtraTreesRegressor`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`

No bearing data or experiment training was run during this environment task.
