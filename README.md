# Multimodal Attendance & Engagement Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Privacy](https://img.shields.io/badge/Privacy-Offline--First-darkgreen.svg)](#privacy--ethical-guardrails)

An **offline-first classroom intelligence application** that records verified session attendance, analyzes observable classroom activity using video and optional audio, generates deterministic pedagogical teaching suggestions, and displays interactive analytics through an educator dashboard.

---

## 🔒 Privacy & Ethical Guardrails

- **No Facial Recognition:** Human presence is tracked strictly via anonymous bounding geometries and integer IDs (`Track-01`, `Track-02`). The system never performs automated facial recognition or biometric identification.
- **Teacher-Confirmed Identity Binding:** Attributing an anonymous track to a student name requires explicit teacher confirmation in the attendance UI or a voluntary student opt-in token.
- **Ephemeral Media Streams:** Raw video frames and raw audio byte arrays are processed transiently in volatile RAM (`np.ndarray`) and discarded immediately. No raw media is written to disk.
- **Observables Over Inferences:** The engine measures physical movement intensity, acoustic volume variance, and speech activity ratios—never purporting to measure internal emotional states or mental comprehension as "ground truth".
- **100% Offline & Local:** Operates without external cloud APIs, telemetry, or remote data egress. All records reside in local SQLite (`attendance_agent.db`).

---

## 📁 Project Directory Structure

```
multimodal-attendance-agent/
│
├── app.py                      # Main Streamlit application entrypoint
├── DEMO_GUIDE.md               # Complete evaluator demonstration guide & test workflows
├── requirements.txt            # Python dependencies (Streamlit, OpenCV, NumPy, Pandas, etc.)
├── README.md                   # Project overview, setup, and instructions
├── .gitignore                  # Git ignore rules (DBs, recordings, caches, venv)
├── requirements.md             # Complete SRS specification and architecture blueprint
│
├── config/
│   └── settings.py             # Centralized typed configuration and paths
│
├── modules/                    # Decoupled core business and perception logic
│   ├── __init__.py             # Public module interface exports
│   ├── pipeline.py             # End-to-end multimodal orchestrator (11-stage pipeline)
│   ├── session_manager.py      # Session lifecycle state machine (READY/ACTIVE/PAUSED/CONCLUDED)
│   ├── video_processor.py      # Video ingestion, OpenCV detection, motion delta & overlays
│   ├── person_tracker.py       # Anonymous CentroidTracker and Track trajectory state
│   ├── attendance.py           # Sustained-presence calculation & roster claims
│   ├── audio_processor.py      # Acoustic feature extraction (RMS, ZCR, Centroid, VAD)
│   ├── feature_fusion.py       # Synchronizes visual and audio metrics into time windows
│   ├── engagement_analytics.py # Observable classroom metrics, distributions & dataframes
│   ├── recommendation_engine.py# Deterministic rule-based pedagogical suggestion engine
│   ├── database.py             # SQLite persistence layer with WAL mode and cascades
│   └── report_generator.py     # Attendance ledger compilation and CSV report exports
│
├── ui/                         # Streamlit presentation views
│   ├── __init__.py
│   ├── dashboard.py            # Main layout container and sidebar navigation
│   ├── home_page.py            # Executive overview, KPI metrics & privacy principles
│   ├── session_page.py         # Live session controls, stream player & batch processing
│   ├── attendance_page.py      # Verified attendance records & teacher manual claim/override
│   ├── analytics_page.py       # Interactive Plotly time-series dynamics charts
│   ├── recommendations_page.py # Rule-based pedagogical suggestions & action plans
│   ├── settings_page.py        # Privacy controls, retention schedules & DB purge tools
│   └── demo_data.py            # Clearly-labelled synthetic demonstration session generator
│
├── tests/                      # Automated unit, integration, and security test suites (138 tests)
│   ├── test_end_to_end_integration.py # Full lifecycle end-to-end integration tests
│   ├── test_privacy_security.py       # Air-gap, zero-media, and path sanitization tests
│   ├── test_qa_edge_cases.py          # Boundary conditions, invalid inputs & cleanup tests
│   ├── test_pipeline.py               # Pipeline orchestrator & batch loop tests
│   ├── test_video_processor.py        # OpenCV frame decoding & detector fallback tests
│   ├── test_audio_processor.py        # Librosa audio features, VAD & chunk processing tests
│   ├── test_person_tracker.py         # Centroid & IoU bounding box tracking tests
│   ├── test_attendance.py             # Sustained presence, transient filtering & threshold tests
│   ├── test_feature_fusion.py         # Temporal window alignment & quality score tests
│   ├── test_analytics.py              # Normalized composite activity indicator tests
│   ├── test_recommendations.py        # Deterministic pedagogical rule trigger tests
│   ├── test_database.py               # SQLite schema, cascades, and transaction tests
│   └── test_ui.py                     # Streamlit component export & demo tour tests
│
├── data/
│   ├── .gitkeep                # Retains local directory in version control
│   └── sample/                 # Sample assets (classroom_demo.mp4, sample_roster.csv)
│
└── docs/
    └── architecture.md         # Comprehensive architectural blueprint and design notes
```

---

## 🚀 Quickstart Guide

> 💡 **Evaluator Quickstart:** For detailed, step-by-step instructions for demonstrating the application (including the offline pre-recorded video pipeline, synthetic demo tour, attendance verification, and CSV exports), refer to [DEMO_GUIDE.md](file:///Users/divyaverma/multimodal-attendance-agent/DEMO_GUIDE.md).

### 1. Prerequisites
- Python 3.10+ (macOS, Linux, or Windows)
- Webcam (optional for live feed; pre-recorded video files supported)
- Microphone (optional; defaults to silent degradation if unavailable)

### 2. Environment Setup
```bash
# Clone the repository
cd multimodal-attendance-agent

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Automated Tests
```bash
pytest -v
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🎙️ Manual Testing Guide: Live Webcam and Microphone

To test the application with real physical hardware:

### Testing the Webcam
1. Connect a USB webcam or ensure the built-in camera is active.
2. In the Streamlit dashboard, click **Classroom Sessions** in the sidebar.
3. In **Input Modality & Source**, select **"Live Camera / Webcam"**.
4. Set **Camera Hardware Index** to `0` (or `1` for external USB webcam).
5. Click **"🔍 Test Camera Stream"**:
   - If the camera is detected, a green confirmation box displays `"Camera 0 opened successfully"` along with the native stream resolution and frame rate.
   - If OS permissions are blocked (e.g., macOS Privacy & Security), an error is shown indicating camera access was denied.
6. Check the **"I acknowledge the Institutional Privacy & Data-Use Protocol"** consent checkbox.
7. Click **"Create Session"**, then **"▶️ Start Session"**.
8. Move across the camera field-of-view:
   - Green bounding boxes with anonymous track IDs (`Track-01`, `Track-02`) will appear over detected individuals.
   - Click **"🔁 Live Playback / Stream"** for continuous real-time updates.

### Testing the Microphone
1. In the **Sensor Modality Controls** section, ensure **"Enable Microphone / Audio Stream Processing"** is checked.
2. Speak near the microphone while the session is active:
   - Observe the live log: `"Microphone stream started successfully at 16000 Hz."`
   - Acoustic features (RMS energy, zero-crossing rate, speech duration ratio) will begin aggregating into the current time window.
3. Navigate to **Analytics** in the sidebar:
   - Verify the **Speech Activity Ratio (%)** chart reflects intervals of speech vs. quiet pauses.
4. Silence the room for > 3 minutes:
   - Navigate to **Teaching Recommendations** and observe if the deterministic rule `EXTENDED_SILENCE` triggers a gentle pedagogical prompt (`"Invite voluntary participant reflections or conduct a quick check-in"`).
5. Return to **Classroom Sessions** and click **"⏹️ Conclude"**:
   - Verify resources are freed: `Video capture resources released cleanly.` and `Audio stream stopped cleanly.`

---

---

## ⚠️ Security and Privacy Limitations

> **Important Disclaimer:** While rigorous static analysis, privacy-by-design patterns, and security code reviews significantly reduce exposure vectors, **a code review or architectural audit cannot guarantee absolute privacy or security**. Institutional operators and educators must understand the inherent limitations of local computer vision and acoustic telemetry:

1. **Host-Level Physical and Terminal Security:**
   - The application relies on the security of the host machine running the Python process. If unauthorized actors gain physical access to the workstation or local terminal, they can inspect local SQLite databases (`data/attendance_agent.db`) or exported CSV reports.
   - Operating system-level disk encryption (e.g., Apple FileVault, Windows BitLocker, Linux LUKS) and strict user permission controls must be enabled on the host machine to protect stored student records at rest.

2. **Transient Memory (RAM) Exposure:**
   - Video frames and audio waveforms are held transiently in volatile system memory (`np.ndarray`) during feature extraction and bounding-box inference before being deallocated.
   - A compromised host operating system with root access, kernel-level debuggers, or unencrypted memory swap files could theoretically capture in-flight memory buffers.

3. **Field-of-View and Acoustic Bleed:**
   - Perception hardware cannot distinguish enrolled participants from casual bystanders or visitors who walk through the camera lens angle or speak near the microphone pickup range.
   - Unintended individuals entering the frame will be assigned transient anonymous tracking IDs. Educators must position cameras and microphones thoughtfully to cover only authorized classroom zones.

4. **Human Verification Integrity:**
   - The system intentionally does **not** perform facial recognition or automated identity attribution.
   - The accuracy of student-to-track identity binding relies entirely on human educator verification during roster matching. Erroneous teacher claims will result in misattributed attendance records.

5. **Hardware Sensor Permission Management:**
   - The application depends on operating system-level hardware access controls (e.g., macOS TCC permissions, Linux device group memberships `/dev/video*`). If permissions are shared or misconfigured at the OS level, external processes could access the sensors concurrently.

6. **Local Network Boundary and Air-Gapping:**
   - The core agent makes zero outbound internet requests. However, when deploying Streamlit over an institutional local area network (LAN), TLS/HTTPS reverse-proxies (e.g., Nginx or Caddy with mutual TLS) and firewall port restrictions (`8501`) must be configured to prevent unauthenticated network eavesdropping.

7. **Institutional Data Retention & Right to Erasure:**
   - Configurable automated purge schedules and manual session deletion tools are provided. However, institutional policies must define retention windows (e.g., 30 or 60 days) to prevent perpetual local storage of historical classroom telemetry.

---

## 🛠️ Current Status & Implementation Roadmap

> **Current Phase:** Production-Ready End-to-End Multimodal Application.
>
> All core architectural modules, database schemas, perception pipelines, analytics, rule engines, UI views, privacy governance guardrails, and automated unit/integration test suites are fully verified and integrated.
