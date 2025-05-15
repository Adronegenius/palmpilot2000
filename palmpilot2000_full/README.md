# PalmPilot 2000

PalmPilot 2000 is a gesture-based 2D layout prototyping tool that combines computer vision and spatial computing for interactive design workflows.

---

## Setup Instructions

1. **Clone the repo** and navigate inside:

```bash
git clone https://github.com/yourusername/palmpilot2000.git
cd palmpilot2000
```

2. **Create a virtual environment**:

```bash
python -m venv venv
```

3. **Activate the environment**:

- macOS/Linux:
```bash
source venv/bin/activate
```

- Windows:
```cmd
venv\Scripts\activate
```

4. **Install the requirements**:

```bash
pip install -r requirements.txt
```

5. **Run a demo**:

```bash
python scripts/main_tracking.py
```

---

## 📂 File Structure

- `scripts/`: Modular FSM scripts and helpers
- `output/`: Saved CSV and PNG layout exports
- `ALL_IN_ONE_version_2.py`: Original monolithic prototype script
