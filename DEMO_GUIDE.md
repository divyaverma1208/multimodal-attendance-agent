# Multimodal Classroom Intelligence Agent — Evaluator Demonstration Guide

This guide provides step-by-step instructions for demonstrating the **Multimodal Attendance & Engagement Agent** to an academic, institutional, or technical evaluator.

The application operates **100% offline**, requiring zero external API keys, zero cloud connectivity, and zero third-party telemetry. Raw video and audio streams are processed ephemerally in RAM and are never persisted to disk.

---

## 🧭 Demonstration Overview & Architecture

The demonstration workflow accommodates three distinct operational environments:

| Demonstration Pathway | Hardware Required | Typical Duration | Focus Areas |
|:---|:---|:---|:---|
| **Route 1: Instant Synthetic Demo Tour** | None (Zero hardware) | 3–5 minutes | Attendance ledger, CSV export, Plotly analytics, pedagogical cues, teacher corrections, and isolation guardrails. |
| **Route 2: Pre-Recorded Video Pipeline** | None (CPU only) | 2–4 minutes | Frame decoding, anonymous person tracking, attendance threshold calculation, fast batch processing, SQLite sync. |
| **Route 3: Live Hardware Processing** | Webcam & Microphone | 3–5 minutes | Live hardware ingestion, anonymous bounding-box overlay, real-time VAD acoustic feature extraction, graceful pause/conclude. |

---

## 🚀 Pre-Demonstration Checklist & Launch

### 1. Verify Environment & Test Suite
Ensure the Python virtual environment is activated and all automated unit tests pass:
```bash
# Activate virtual environment
source .venv/bin/activate

# Execute all 138 unit tests (must pass 100%)
pytest -q
```
*Expected Output:* `138 passed in ~5s`.

### 2. Launch the Streamlit Application
Start the Streamlit dashboard on `localhost`:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in any modern web browser.

---

## 🧪 Route 1: One-Click Synthetic Demo Tour (Zero Hardware Required)

This route demonstrates the full reporting and analytical capabilities of the agent using rich, clearly labelled synthetic data.

### Step 1.1: Load Synthetic Demo Data
1. Navigate to the **Home** page or **Classroom Sessions** page in the left sidebar.
2. In the **Synthetic Demo Tour Mode** section, click:
   $$\boxed{\text{📥 Load Synthetic Sample Session [DEMO]}}$$
3. **Observation:** A success banner confirms that session `[DEMO] Intro to Computer Systems (Synthetic Sample)` has been populated in the local SQLite database.

> [!NOTE]
> **Synthetic Data Isolation Guardrail (Requirements 2 & 3):**
> Synthetic data is explicitly prefixed with `[DEMO]` in session titles, table listings, and alert banners. Synthetic records are never masqueraded as live model observations.

---

### Step 1.2: Inspect Attendance Verification & Attendance Events
1. In the sidebar, click on **📋 Attendance**.
2. Notice the prominent amber banner at the top:
   > 🧪 **Notice: [DEMO / SYNTHETIC DATA]** — This session contains pre-generated synthetic demo records. It is strictly isolated and never confused with live sensor output.
3. Observe the four high-level KPI cards:
   - **Total Observed Tracks:** `6`
   - **Verified Present:** `3` (satisfying the 60% sustained presence threshold)
   - **Provisional / In-Progress:** `1`
   - **Partial / Transient:** `2`

4. Inspect the **Three Tabbed Ledgers**:
   - **Tab 1: `✅ Verified Attendance (3)`**: Displays confirmed attendees (e.g. Alice Chen, Bob Vance, and an unassigned anonymous sustained track).
   - **Tab 2: `⏳ Provisional & In-Progress (3)`**: Displays provisional observations, partial attendees (e.g., student excused early), and transient passers-by (filtered under 30s).
   - **Tab 3: `📜 Attendance Events Log (3)`**: Displays the chronological event stream (`ENTRY`, `SUSTAINED_PRESENCE_MET`, `MANUAL_CORRECTION`) showing relative timestamps, anonymous track IDs, and descriptive audit logs.

---

### Step 1.3: Demonstrate CSV Export (Requirement 8)
1. In the **Verified Attendance** tab, click:
   $$\boxed{\text{📥 Download Verified Attendance CSV}}$$
   - Opens or saves `verified_attendance_CS101_<session_id>.csv`.
   - Contains enrolled student IDs, verified student names, sustained presence flags, and total active minutes.
2. In the **Provisional & In-Progress** tab, click:
   $$\boxed{\text{📥 Download Provisional Audit CSV}}$$
   - Contains tentative observations for audit verification.
3. In the **Attendance Events Log** tab, click:
   $$\boxed{\text{📥 Download Attendance Events Log CSV}}$$
   - Contains the timestamped perception event log.

---

### Step 1.4: Demonstrate Teacher Manual Overrides & Roster Binding
1. Scroll down to the **✍️ Teacher Manual Corrections & Roster Binding** section.
2. **Assign Student:**
   - Under *Link Track to Enrolled Student*, select `Track-3 [SUSTAINED_PRESENT]`.
   - Select an enrolled student from the roster: `Diana Prince (STU-104)`.
   - Click **Confirm Student Assignment**.
   - **Observation:** Track-3 is immediately bound to Diana Prince, updating the ledger with audit logging.
3. **Manual Status Override:**
   - Under *Manual Status Override*, select any track.
   - Change status (e.g., override a provisional track to `SUSTAINED_PRESENT` with note `"Teacher verified via roll call"`).
   - Click **Apply Status Override**. The database ledger and attendance events log update instantaneously.

---

### Step 1.5: Inspect Activity Dynamics & Sensor Gaps (Requirements 6 & 9)
1. In the sidebar, click on **📊 Analytics**.
2. Notice the **Sensor Availability & Data Quality** indicators:
   - Evaluates camera coverage, microphone coverage, and high-quality window percentages.
   - Highlights simulated sensor dropouts (e.g., `"1 window(s) missing audio input (shown as gaps, NOT zero speech)"`).
3. View the **Interactive Plotly Time-Series Visualizations**:
   - **Trace 1 (Green line):** Speech Activity Ratio (%) across 30-second windows.
   - **Trace 2 (Blue dashed line):** Visual Movement Index (%).
   - **Subplot 2 (Amber bars):** Observed Attendee Headcount.
   - **Discontinuity Handling:** Notice how missing sensor windows appear as clean gaps rather than false $0.0$ drops.
4. Review **Dynamic Shifts & Instructional Profile**:
   - Displays macro phase transitions (e.g., ramp-up from monologue to group discussion).
   - Quantifies estimated lecture monologue minutes vs. interactive dialogue minutes.

---

### Step 1.6: Inspect Teaching Recommendations (Requirement 7)
1. In the sidebar, click on **💡 Teaching Recommendations**.
2. View generated pedagogical decision-support cues:
   - `📌 RULE_PROLONGED_MONOLOGUE`: Triggered when speech ratio exceeds 80% for $\ge 10$ minutes. Suggests a quick recap or peer Q&A.
   - `📌 RULE_RESTLESSNESS_SPIKE`: Triggered when movement spikes while speech is low. Suggests a 60-second stretch or transition.
3. **Interactive Acknowledgment:**
   - On a pending suggestion, click **Mark Acknowledged**.
   - **Observation:** Status updates to `✅ Acknowledged` with timestamped audit logging.
4. **Inspect Rule Engine Inventory:**
   - Scroll down to review the table of all 8 deterministic, multi-window pedagogical heuristics.

---

### Step 1.7: Display the Limitations of Automated Observations (Requirement 9)
Demonstrate the epistemic modesty disclosures on the **Analytics** and **Teaching Recommendations** pages:
1. On **Analytics**, view the dedicated section: **⚠️ Limitations of Automated Classroom Observation**:
   - **Physical Proxies Only:** Explains that movement index and speech ratio reflect gross physical movement and acoustic energy only—never internal attention, emotion, engagement, cognitive comprehension, or intelligence.
   - **Sensor Outages vs. Zeros:** Re-iterates that camera occlusion or muted microphones are recorded as `Unavailable`, not zero.
   - **Strict Non-Punitive Standard:** Explains that automated observation metrics must never be used for disciplinary action, punitive grading, or evaluating teacher competence.
   - **Human Primacy:** The teacher retains complete authority to override or discard algorithmic suggestions.
2. On **Teaching Recommendations**, view **🛡️ Ethical Boundaries & Observation Limitations**:
   - Confirms recommendations are optional, voluntary decision-support tools.
   - Confirms the rule engine is deterministic without opaque black-box machine learning models.

---

## 📹 Route 2: Pre-Recorded Video Pipeline Demonstration (`classroom_demo.mp4`)

This route demonstrates the live computer vision and tracking pipeline running against a bundled, offline MP4 video (`data/sample/classroom_demo.mp4`).

### Step 2.1: Initialize Session with Sample Video
1. In the sidebar, click on **📹 Classroom Sessions**.
2. Under **Initialize New Classroom Session**:
   - **Course Code:** Enter `DEMO101`.
   - **Session Title:** Enter `Offline Video Demonstration`.
   - **Video Input Source Mode:** Select `Recorded Video File`.
   - **Video File Method:** Select `Sample Demo Video`.
   - Notice the confirmation badge: `Selected pre-recorded sample video: data/sample/classroom_demo.mp4 (10s offline video, 3 tracked participants, 10 FPS)`.
3. **Sensor Controls:** Leave *Enable Camera Stream Processing* checked. Set *Target Processing Frame Rate* to `10 FPS` and *Sustained Presence Threshold* to `50%`.
4. **Institutional Notice:** Review the Institutional Privacy Notice and check:
   $$\boxed{\checkmark \text{ I acknowledge the Institutional Privacy & Data-Use Protocol...}}$$
5. Click **Create Session**.

---

### Step 2.2: Live Video Playback & Tracking Overlays
1. Once created, the session appears in the `READY` state. Click **▶️ Start Session**.
2. Observe the real-time perception feed:
   - Green bounding boxes track individual students: `Person 1`, `Person 2`, `Person 3`.
   - Red centroids mark tracking focal points.
   - Watermark displays resolution, effective processing FPS, active head count, and motion index.
3. Test Controls:
   - Click **⏭️ Next Frame** to advance frame-by-frame.
   - Check **🔁 Live Playback / Stream** for smooth continuous playback.
   - Notice how Student 1 and Student 2 remain seated (sustained presence), while Student 3 departs halfway through the video (partial/transient observation).

---

### Step 2.3: Fast Batch Processing & Session Conclusion (Requirement 4)
1. Click the button:
   $$\boxed{\text{⚡ Fast-Process Entire Video}}$$
2. **Observation:** A progress bar processes all 100 frames through the detection, tracking, feature fusion, and database pipeline in ~2 seconds.
3. The session automatically concludes, releasing all OpenCV VideoCapture handles and persisting evaluations to SQLite.
4. Navigate to **Attendance** to verify that Track-1, Track-2, and Track-3 have been recorded with exact entry timestamps and duration metrics!

---

## 🎙️ Route 3: Live Hardware Webcam & Microphone Demonstration

*(Optional: Use if the host machine has an accessible camera and microphone.)*

1. Navigate to **Classroom Sessions**.
2. Under **Video Input Source Mode**, select `Live Webcam`.
3. Set **Camera Device Index** to `0` (or `1` for USB webcam).
4. Click **🔍 Test Camera** to confirm device accessibility and resolution.
5. Check the institutional privacy consent box and click **Create Session** -> **▶️ Start Session**.
6. **Live Perception:**
   - Person tracker draws anonymous green bounding boxes over attendees in real-time.
   - Microphone extracts short-time RMS energy and speech ratio without saving audio.
7. Test Hardware Toggles:
   - Disable camera or microphone in settings; notice how missing modalities are logged as `Unavailable` rather than crashing the pipeline.
8. Click **⏹️ Conclude** to safely release hardware handles and commit records.

---

## 🛡️ Privacy, Data Retention & Permanent Deletion

1. In the sidebar, navigate to **⚙️ Privacy and Settings**.
2. **Review Privacy Assertions:**
   - Visual and acoustic processing occurs 100% locally in volatile RAM.
   - Zero raw video or audio recordings are saved to disk.
   - Database and temporary caches are strictly excluded from version control (`.gitignore`).
3. **Configurable Data Retention (Requirement 6):**
   - Select retention policy (e.g., 30 days, 90 days, or manual purge).
   - Click **Run Database Retention Purge** to clean expired records.
4. **Permanent Session Deletion:**
   - Under *Permanent Session Deletion*, select any session.
   - Confirm acknowledgment and click **Permanently Delete Session**.
   - **Observation:** Session metadata, attendance records, events, activity windows, and suggestions are purged permanently via SQL cascade.
5. **Clean Up Demo Data:**
   - Click **🗑️ Clean Up All Demo Data** to remove all `[DEMO]` sessions with one click.

---

## 📋 Evaluator Requirements Verification Matrix

| Requirement | Implementation Feature | Verification Step in Demo |
|:---|:---|:---|
| **1. Recorded-video demo option** | Bundled `data/sample/classroom_demo.mp4` with 3 tracked students | Select `Recorded Video File` -> `Sample Demo Video` in Sessions page. |
| **2. Clearly labelled demo-data option** | One-click synthetic session generator in `ui/demo_data.py` | Click `📥 Load Synthetic Sample Session [DEMO]` on Home or Sessions page. |
| **3. Separation of demo data vs. live** | `[DEMO]` prefix, amber warning banners, distinct labeling | View Attendance/Analytics pages for `[DEMO]` sessions; notice persistent warnings. |
| **4. Session creation & completion** | Full lifecycle in `modules/session_manager.py` & `ui/session_page.py` | Create session -> Start -> Step/Stream -> Conclude (or `⚡ Fast-Process`). |
| **5. Attendance events log** | `db.get_attendance_events()` table in `ui/attendance_page.py` | View Tab 3: `📜 Attendance Events Log` on Attendance page. |
| **6. Activity trends** | Plotly multi-trace time-series in `ui/analytics_page.py` | View speech ratio %, movement %, and headcount trends over time. |
| **7. Teaching recommendations** | Transparent rule engine in `modules/recommendation_engine.py` | View `💡 Teaching Recommendations` page; acknowledge pending suggestions. |
| **8. CSV export** | RFC 4180 CSV generators in `modules/report_generator.py` | Click `Download Verified Attendance CSV` & `Download Attendance Events Log CSV`. |
| **9. Observation limitations** | Epistemic modesty sections in UI & settings | Review `⚠️ Limitations of Automated Classroom Observation` on Analytics page. |
| **10. 100% Offline execution** | Local OpenCV, VAD, SQLite, zero cloud APIs | Disconnect WiFi/network; app continues to function flawlessly. |
