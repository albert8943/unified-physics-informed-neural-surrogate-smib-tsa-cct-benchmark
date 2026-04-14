# Running the complete experiment

## Blank PowerShell terminal (no prompt, only blinking cursor)

If the integrated terminal opens but stays **blank** (no path, no prompt), try:

### 1. Use a different terminal type

- In the terminal panel, click the **dropdown** next to the **+** (or the **∨** next to the current terminal name).
- Choose **Command Prompt** or **Git Bash** instead of PowerShell.
- Open a **new** terminal (e.g. **Terminal → New Terminal** or `` Ctrl+Shift+` ``).

You should see a normal prompt (e.g. `D:\...\PINN_project_using_andes>`). Then run your commands there (use `conda activate pinn_andes_1.10` first if you use conda).

### 2. Make Command Prompt the default (so new terminals are never blank)

- Press `Ctrl+Shift+P` → **Preferences: Open User Settings (JSON)**.
- Add (or merge) this so **Command Prompt** is used by default on Windows:

```json
"terminal.integrated.defaultProfile.windows": "Command Prompt"
```

- Save, then close all terminals and open a new one (**Terminal → New Terminal**).

### 3. If you must use PowerShell

- Same Settings (JSON), try turning shell integration off (sometimes fixes blank output):

```json
"terminal.integrated.shellIntegration.enabled": false
```

- Or set the full path to PowerShell so a clean instance is used:

```json
"terminal.integrated.defaultProfile.windows": "PowerShell",
"terminal.integrated.profiles.windows": {
  "PowerShell": {
    "path": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    "args": ["-NoLogo"]
  }
}
```

### 4. Conda terminals showing a warning

If **conda** terminals show a warning icon and don’t work, **don’t** choose “Conda” as the terminal type. Use **Command Prompt** (or PowerShell once it’s fixed), then run:

```bat
conda activate pinn_andes_1.10
python scripts/run_complete_experiment.py --config configs/publication/smib_delta20_omega40.yaml --skip-data-generation --data-dir data/processed/exp_20260204_190157
```

---

## Terminal error: `ModuleNotFoundError: No module named 'pandas'`

This happens when **the wrong Python is used** (e.g. system Python 3.13 instead of your project environment).

### Fix in Cursor

1. **Select the correct Python interpreter**
   - Press `Ctrl+Shift+P` → type **"Python: Select Interpreter"** → Enter.
   - Choose the environment where you installed project dependencies, e.g.:
     - **`pinn_andes_1.10`** (conda), or
     - A **venv** in this project (e.g. `./.venv` or `./venv`).
   - If you don’t see it, choose **"Enter interpreter path..."** and point to that environment’s `python.exe` (e.g. `C:\Users\Albert\anaconda3\envs\pinn_andes_1.10\python.exe` or your venv’s `Scripts\python.exe`).

2. **Use the integrated terminal**
   - Open terminal in Cursor: **View → Terminal** or `` Ctrl+` ``.
   - The terminal should now use the selected interpreter. Run:
     ```powershell
     python scripts/run_complete_experiment.py --config configs/publication/smib_delta20_omega40.yaml --skip-data-generation --data-dir data/processed/exp_20260204_190157
     ```

3. **Or run without terminal (Run and Debug)**
   - Open `scripts/run_complete_experiment.py`.
   - Right‑click in the editor → **Run Python File in Terminal** (uses the selected interpreter).

### If you use conda

In a **new** terminal, activate first, then run:

```powershell
conda activate pinn_andes_1.10
python scripts/run_complete_experiment.py --config configs/publication/smib_delta20_omega40.yaml --skip-data-generation --data-dir data/processed/exp_20260204_190157
```

### If you use a venv

```powershell
.\.venv\Scripts\Activate.ps1
# or: .\venv\Scripts\Activate.ps1
python scripts/run_complete_experiment.py --config configs/publication/smib_delta20_omega40.yaml --skip-data-generation --data-dir data/processed/exp_20260204_190157
```

Once the **correct** interpreter (the one with pandas, torch, etc.) is selected or activated, the `ModuleNotFoundError` should go away.
