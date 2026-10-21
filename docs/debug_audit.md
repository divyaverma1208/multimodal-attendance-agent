# Comprehensive Debug Audit & Root Cause Analysis
**Project:** Classroom Agent — Multimodal Attendance & Engagement  
**Document:** `docs/debug_audit.md`  
**Date:** October 20, 2026  
**Auditor:** Antigravity Senior Engineering & QA Team  

---

## Executive Summary

A comprehensive code inspection and data flow trace was conducted across all 12 modules, configuration settings, SQLite database persistence layer, Streamlit UI controllers, and 145 automated test suites. 

While the high-level architecture and component boundaries are well-structured, several critical bugs and interface mismatches were identified that directly explain why person tracking active duration was stuck at 0 seconds, speech ratio was stuck on "Awaiting audio", the activity index remained "Pending Window", and dashboard updates caused screen flickering.

This audit documents each observed issue, the exact file and function, the verified root cause supported by code inspection, the proposed fix, and the verification test required.

---

## Data Flow Pipeline Trace

```mermaid
flowchart TD
    subgraph Input_Layer [Sensor Ingestion]
        CAM[OpenCV Camera / Video File] --> VP[VideoProcessor]
        MIC[SoundDevice Microphone / Audio File] --> AP[AudioProcessor]
    end

    subgraph Perception_Layer [Feature Extraction]
        VP --> DET[Face / Body / HOG Detectors]
        DET --> NMS[Dual-Metric NMS (IoU + IoM)]
        NMS --> TRK[CentroidTracker (Tracking IDs)]
        VP --> MOT[Frame Differencing Motion Index]
        AP --> VAD[Multi-Feature VAD (RMS + ZCR + Spectral Centroid)]
    end

    subgraph Fusion_Layer [Temporal Synchronization]
        TRK --> ATT[AttendanceEngine (Sustained Observations)]
        MOT & VAD --> FFE[FeatureFusionEngine (30s Time Windows)]
    end

    subgraph Intelligence_Layer [Analytics & Pedagogical Support]
        FFE --> ENG[EngagementAnalytics (Composite Activity Score)]
        FFE --> REC[RecommendationEngine (Rule-Based Decision Support)]
    end

    subgraph Persistence_Layer [Local Storage & Export]
        ATT & FFE & REC --> DB[(Local SQLite Database - WAL Mode)]
        DB --> CSV[ReportGenerator (Attendance CSV Export)]
    end

    subgraph Presentation_Layer [Streamlit Dashboard]
        VP & TRK & FFE & REC --> UI[In-Place Telemetry & Live Video Slots]
    end
```

---

## Issue Audit Matrix

| Issue ID | Domain | Observed Symptom | File & Function | Root Cause | Proposed Fix | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AUD-01** | **Tracking** | Track observed frames stuck at 1, active duration 0s, ID churn | `modules/person_tracker.py` -> `CentroidTracker.update`<br>`modules/video_processor.py` -> `VideoProcessor.process_frame` | Centroid distance threshold (75px) was exceeded by minor head movement or detector box scale changes; lack of IoU/overlap matching in association; timestamps were not passed into `process_frame`. | Add IoU and containment overlap to association; adapt centroid tolerance to bbox size; pass elapsed timeline timestamp into `process_frame`. | `tests/test_person_tracker.py` (multi-frame track ID stability, duration > 0s) |
| **AUD-02** | **Audio** | Speech ratio stuck on "Awaiting audio" / Method & Attribute Mismatches | `modules/pipeline.py` -> `ClassroomPipeline.step` (line 294)<br>`modules/audio_processor.py` -> `AudioFeatures`<br>`ui/session_page.py` -> `update_telemetry_ui` (line 570) | 1. `pipeline.py` called `audio_processor.process_audio_chunk()` which did not exist on `AudioProcessor` (name was `extract_features()`), caught silently as `AttributeError`.<br>2. `ui/session_page.py` accessed `audio_features.speech_ratio` instead of `speech_ratio_pct`. | 1. Add `process_audio_chunk = extract_features` alias.<br>2. Add `@property speech_ratio` to `AudioFeatures`.<br>3. Handle both ratio and percentage gracefully in UI. | `tests/test_audio_processor.py` & `tests/test_pipeline.py` (mic chunk read & speech ratio display) |
| **AUD-03** | **Analytics** | Activity Index stuck on "Pending Window" | `modules/pipeline.py` -> `ClassroomPipeline.step`<br>`ui/session_page.py` -> `update_telemetry_ui` | `composite_activity` was only returned on the single frame where a 30s window closed; on subsequent frames it was `None`, immediately resetting UI to "Pending Window". | Cache `latest_composite_activity` in pipeline; report window accumulation progress (`Accumulating Xs/30s`) during interim periods. | `tests/test_feature_fusion.py` & `tests/test_analytics.py` (persistent score & window progress) |
| **AUD-04** | **Audio** | Hardware microphone access failure reported as operational | `modules/audio_processor.py` -> `AudioProcessor.start_mic_stream`<br>`ui/dashboard.py` -> status indicator | Dashboard reported `Acoustic VAD: Ready (Local)` statically without checking if physical microphone device stream succeeded. | Query `AudioProcessor.status` dynamically; display clear warning if microphone is unavailable; provide recorded audio test mode. | `tests/test_audio_processor.py` (device unavailable error handling) |
| **AUD-05** | **Session** | Dangling ACTIVE sessions and duplicate session starts | `modules/session_manager.py` -> `SessionManager.create_session`, `start_session` | No check for pre-existing active sessions in DB upon startup; crash or browser close left unclosed ACTIVE sessions in SQLite. | Add `recover_unclosed_sessions()` on startup; enforce strict state transitions and prevent duplicate concurrent active sessions. | `tests/test_qa_edge_cases.py` & `tests/test_database.py` (state transitions & recovery) |
| **AUD-06** | **Attendance** | Single-frame walk-in risk & missing duration synchronization | `modules/attendance.py` -> `AttendanceEngine.evaluate_session_presence` | If session concluded before tracks were flushed, unobserved or transient tracks could have invalid elapsed ratios. | Ensure `evaluate_session_presence` strictly filters transient tracks (<30s) and requires sustained presence (>=60% of active time). | `tests/test_attendance.py` (transient filter and sustained presence verification) |
| **AUD-07** | **Recommendations** | Suggestions not updating or failing on missing modalities | `modules/recommendation_engine.py` -> `evaluate_rules` | Rules required 2 complete windows; because windows were resetting or audio was failing, recommendations remained dormant. | Connect recommendation engine to cached multimodal window history; handle audio-only, video-only, and multimodal inputs cleanly. | `tests/test_recommendations.py` (rule evaluations under partial modalities) |
| **AUD-08** | **Dashboard** | Screen flickering and UI state resets | `ui/session_page.py` -> continuous streaming loop | Per-frame `st.rerun()` reloaded entire Streamlit DOM; replaced with in-place placeholder slot updates with rate pacing. | Use in-place `empty().image()` and `slot.metric()` within smooth batch streaming loop (15 frames/burst @ ~25 FPS). | Manual browser validation & `tests/test_ui.py` |
| **AUD-09** | **Privacy** | Overstated compliance claim in dashboard banner | `ui/dashboard.py` -> `render_privacy_banner` | Banner claimed "AIR-GAPPED COMPLIANT" without rigorous local proof. | Update wording to "LOCAL OFFLINE OPERATION • ZERO CLOUD EGRESS • AIR-GAP READY"; verify zero network calls and git exclusions. | `tests/test_privacy_security.py` (network isolation & git exclusion verification) |

---

## Detailed Root Cause Analysis by Component

### 1. Person Tracking & Identity Management (`modules/person_tracker.py`, `modules/video_processor.py`)
- **Problem:** Users reported that when one person stood in front of the camera, the system frequently assigned new IDs (`Track 1`, `Track 2`, `Track 3`), keeping frames observed at 1 and active duration at 0s.
- **Root Cause:**
  - `CentroidTracker` matched existing tracks exclusively by Euclidean distance `math.hypot(ex - cx, ey - cy) <= self.max_distance_px` (75.0 px).
  - In a standard 640x480 or 1280x720 video feed, human movement or detector box jitter (e.g. face detected then upper-body detected) moves the centroid by 80–120 pixels.
  - When this occurred, the greedy distance matching rejected the pairing. The existing track was marked `missing`, while the detection was registered as a new track ID with `total_frames_detected = 1` and `first_seen_timestamp = now`.
  - Furthermore, `process_frame` did not always receive explicit frame timestamps, falling back to `time.time()`. For pre-recorded video files processed in fast mode, wall-clock time advanced only milliseconds, giving 0.0s elapsed active time.
- **Solution:**
  - Implement dual-metric matching: compute both Euclidean centroid distance and Bounding Box IoU / IoM overlap.
  - If IoU > 0.20 or centroid is inside the previous bounding box, preserve the track ID regardless of pixel displacement.
  - Normalize distance threshold relative to bounding box diagonal (e.g. `max(75.0, 0.5 * bbox_diag)`).
  - In `VideoProcessor.process_frame`, compute the exact timeline timestamp (`frame_index / source_fps`) for recorded video or elapsed monotonic time for webcams.

### 2. Audio Processing & Voice Activity Detection (`modules/audio_processor.py`, `modules/pipeline.py`)
- **Problem:** Dashboard reported `Speech Ratio: Awaiting audio` indefinitely even with an active microphone.
- **Root Cause:**
  - In `modules/pipeline.py` line 294:
    ```python
    audio_features = self.audio_processor.process_audio_chunk(chunk, timestamp=now)
    ```
    `AudioProcessor` has no method named `process_audio_chunk`. The actual method is `extract_features(audio_chunk, timestamp)`. Because this call was wrapped in a broad `try...except`, an `AttributeError` was caught and silently logged at debug level, setting `audio_features = None` on every single iteration!
  - In `ui/session_page.py` line 570:
    ```python
    sr = latest_res.audio_features.speech_ratio if (latest_res and latest_res.audio_features) else None
    ```
    The dataclass `AudioFeatures` defines `speech_ratio_pct: float` (0.0 to 100.0), NOT `speech_ratio`. Thus `latest_res.audio_features.speech_ratio` did not exist.
- **Solution:**
  - Add `process_audio_chunk = extract_features` alias to `AudioProcessor`.
  - Add `@property speech_ratio(self) -> float: return self.speech_ratio_pct / 100.0` to `AudioFeatures`.
  - Handle both properties gracefully in `ui/session_page.py`.

### 3. Feature Fusion & Analytics (`modules/feature_fusion.py`, `modules/pipeline.py`)
- **Problem:** `Activity Index` remained stuck on `Pending Window`.
- **Root Cause:**
  - In `ClassroomPipeline.step()`, `composite_score` was only calculated when `fusion_engine.is_window_ready(now)` returned `True` (once every 30 seconds).
  - On the very next frame, `completed_window` was `None` and `composite_score` was `None`.
  - In `ui/session_page.py`, when `step_res.composite_activity is None`, the metric was overwritten with `"Pending Window"`.
  - Thus, the activity score was only rendered for a single 40ms frame every 30 seconds, immediately reverting to "Pending Window".
- **Solution:**
  - Cache `latest_composite_activity` on `ClassroomPipeline` so every subsequent step retains the most recently fused activity score.
  - Provide a window accumulation progress indicator (e.g. `Accumulating (14s/30s)`) while the initial or current window is filling.

### 4. Session State & Database Integrity (`modules/session_manager.py`, `modules/database.py`)
- **Problem:** Unexpected terminations left sessions in `ACTIVE` state in SQLite, and duplicate sessions could be created.
- **Root Cause:**
  - `SessionManager` lacked an automatic startup scan to detect unclosed sessions from previous application crashes.
  - `create_session` did not guard against an already running session.
- **Solution:**
  - Implement `recover_or_conclude_stale_sessions()` in `SessionManager` that marks dangling `ACTIVE` sessions as `CONCLUDED` with proper duration accounting upon startup.
  - Disallow creating a new session while another session is in `ACTIVE` state.

---

## Conclusion & Action Plan

With all root causes identified and confirmed against actual code implementations, we proceed to Phase 2 through Phase 9 in systematic order:
1. **Phase 2:** Person tracking stabilization (dual IoU/centroid association, timeline timestamps, track persistence).
2. **Phase 3:** Audio pipeline fix (`process_audio_chunk`, `speech_ratio` property, hardware status & recorded audio testing).
3. **Phase 4:** Feature fusion & activity index caching (window progress, persistent metrics, multimodal and single-modality handling).
4. **Phase 5:** Session lifecycle hardening (duplicate prevention, stale session recovery, database persistence).
5. **Phase 6:** Teaching recommendation engine verification.
6. **Phase 7:** Dashboard state & smooth streaming validation.
7. **Phase 8:** Privacy verification & security checks.
8. **Phase 9:** Full test suite execution and integration testing.
9. **Phase 10:** Final Acceptance Report.
