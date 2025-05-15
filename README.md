# PalmPilot 2000

PalmPilot 2000 is a computer vision-based spatial layout tool that enables users to interactively arrange 2D elements using hand gestures. Developed for the MRAC Hardware seminar, the system combines a finite state machine (FSM) with real-time gesture recognition to provide an intuitive and touchless design interface.

## Project Overview

This project explores body-centered interaction in spatial computing. Instead of relying on complex UI or traditional CAD software, users can move, rotate, and place layout elements simply by gesturing in front of a camera. This makes it ideal for early-stage schematic design, educational demonstrations, or user-centered prototyping in architecture and design workflows.

The project integrates a Python-based FSM system with MediaPipe for gesture recognition, OpenCV for computer vision rendering, and UDP communication with Rhino/Grasshopper for live data transfer.

## Features

- Real-time hand gesture tracking using MediaPipe
- Finite State Machine (FSM) for handling interactive states (pick, move, rotate, save, reset)
- Visual overlay of blocks in a defined workspace using OpenCV
- Exports layout data as CSV and image for documentation or further use in design tools
- Communicates with Grasshopper via UDP for hybrid workflows

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/palmpilot2000.git
   cd palmpilot2000
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```cmd
     venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the main script:**
   ```bash
   python scripts/main_tracking.py
   ```

## File Structure

- `scripts/` — Contains modular FSM, vision, and calibration scripts
- `layouts/` — Includes example Rhino (`.3dm`) and Grasshopper (`.gh`) files for integration
- `output/` — Exports of layout data in PNG and CSV formats
- `ALL_IN_ONE_version_2.py` — Original prototype script containing the full logic in one file
- `requirements.txt` — All required Python packages
- `README.md` — This file

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

---

© 2025 PalmPilot 2000 was created as part of the MRAC Hardware seminar coursework and remains an open tool for experimentation and design education.
