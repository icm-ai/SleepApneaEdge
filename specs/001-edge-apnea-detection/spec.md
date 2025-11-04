# Feature Specification: Edge-Based Sleep Apnea Detection

**Feature Branch**: `001-edge-apnea-detection`
**Created**: 2025-11-03
**Status**: Draft
**Input**: User description: "edge-apnea-detection"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Real-Time Apnea Event Detection (Priority: P1)

A person with suspected sleep apnea uses a wearable device during sleep. The device monitors their breathing patterns and detects apnea events (pauses in breathing) in real-time without requiring internet connectivity or cloud processing.

**Why this priority**: This is the core value proposition - detecting apnea events as they happen to enable immediate alerts or interventions. This forms the minimum viable product.

**Independent Test**: User wears the device overnight. The system detects and logs apnea events. Next morning, user reviews detected events with timestamps and severity indicators. Can be fully tested with one night of sleep data.

**Acceptance Scenarios**:

1. **Given** the user is wearing the device during sleep and experiencing normal breathing, **When** breathing patterns are analyzed, **Then** no apnea events are flagged
2. **Given** the user experiences a breathing pause ≥10 seconds (clinical apnea definition), **When** the event occurs, **Then** the system detects and logs it within 2 seconds with timestamp and duration
3. **Given** multiple apnea events occur during one sleep session, **When** the session ends, **Then** all events are stored locally with accurate timestamps and durations
4. **Given** the device has limited battery power, **When** running detection for 8 hours, **Then** battery consumption remains within acceptable limits for overnight use

---

### User Story 2 - Apnea Event Classification (Priority: P2)

Users and healthcare providers need to understand the type and severity of detected apnea events to inform treatment decisions. The system classifies events as obstructive, central, or mixed apnea, and calculates an Apnea-Hypopnea Index (AHI) score.

**Why this priority**: Classification adds clinical value beyond basic detection, helping differentiate between apnea types that require different treatments. This enhances diagnostic utility.

**Independent Test**: Using validated sleep study data with known apnea types, verify the system correctly classifies event types with ≥90% accuracy and calculates AHI scores within ±2 events/hour of clinical standards.

**Acceptance Scenarios**:

1. **Given** an obstructive apnea event (airflow stops but breathing effort continues), **When** analyzed, **Then** system classifies it as "obstructive" with confidence score
2. **Given** a central apnea event (both airflow and breathing effort stop), **When** analyzed, **Then** system classifies it as "central" with confidence score
3. **Given** 8 hours of sleep data with detected events, **When** session analysis completes, **Then** system calculates AHI score (events per hour) accurate to within ±2 events/hour
4. **Given** mixed apnea patterns during sleep, **When** classification runs, **Then** each event receives appropriate type label with ≥90% accuracy vs clinical ground truth

---

### User Story 3 - Historical Trend Analysis (Priority: P3)

Users want to track their sleep apnea patterns over weeks or months to understand if their condition is improving, worsening, or stable. This helps evaluate treatment effectiveness.

**Why this priority**: Trend analysis provides longitudinal insights valuable for treatment monitoring, but the system delivers value even without this feature via real-time detection.

**Independent Test**: After 30 nights of recorded sleep data, user views trends showing AHI scores over time, event frequency patterns, and severity distribution. Trends accurately reflect stored session data.

**Acceptance Scenarios**:

1. **Given** 30 nights of sleep data stored locally, **When** user requests trend view, **Then** system displays AHI scores per night as a time-series graph
2. **Given** historical data showing improvement (decreasing AHI), **When** trend analysis runs, **Then** system highlights positive trend with clear visualization
3. **Given** user wants to identify patterns, **When** viewing weekly summaries, **Then** system shows average events per night, worst night, and best night metrics
4. **Given** storage constraints on edge device, **When** 90+ nights of data accumulate, **Then** system manages storage by archiving or summarizing older data while preserving key metrics

---

### Edge Cases

- What happens when sensor data quality is poor (e.g., device not worn properly)?
  - System should detect low signal quality and alert user rather than producing false detections
- How does the system handle borderline events (breathing pauses of 8-9 seconds, just below 10-second threshold)?
  - Log as "hypopnea" events separately, include in AHI calculation with appropriate weighting
- What if the user has other sleep-related breathing disorders (e.g., Cheyne-Stokes respiration)?
  - System should detect and flag atypical patterns even if they don't fit standard apnea classifications
- How does the system perform during daytime testing or short naps (<2 hours)?
  - Detection works on any duration, but AHI calculations require minimum session length (document threshold)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST detect breathing pauses ≥10 seconds during sleep sessions
- **FR-002**: System MUST log each detected event with timestamp, duration, and estimated severity
- **FR-003**: System MUST perform all detection and analysis on the edge device without requiring cloud connectivity
- **FR-004**: System MUST process sensor data in real-time with ≤100ms latency per analysis window
- **FR-005**: System MUST store at least 90 nights of sleep session data on device
- **FR-006**: System MUST classify apnea events into obstructive, central, or mixed types with ≥90% accuracy
- **FR-007**: System MUST calculate Apnea-Hypopnea Index (AHI) score per sleep session
- **FR-008**: System MUST distinguish between apnea events (≥10s pauses) and hypopnea events (reduced airflow)
- **FR-009**: System MUST detect and flag low-quality sensor data rather than producing false positives
- **FR-010**: System MUST provide session summaries showing total events, AHI score, event type distribution
- **FR-011**: System MUST display historical trends across multiple sleep sessions
- **FR-012**: System MUST operate continuously for 8-10 hours on battery power during typical sleep session
- **FR-013**: System MUST allow export of sleep data in standard medical reporting format for healthcare provider review

### Key Entities

- **Sleep Session**: A continuous period of monitoring (typically 6-10 hours), including start time, end time, total duration, and session quality score
- **Apnea Event**: A detected breathing pause, including timestamp, duration, type (obstructive/central/mixed), confidence score, and associated sensor readings
- **Hypopnea Event**: A detected reduction in airflow, including timestamp, duration, severity percentage, and confidence score
- **AHI Score**: Calculated metric representing events per hour, including breakdown by event type and severity classification (normal: <5, mild: 5-15, moderate: 15-30, severe: >30)
- **Device Profile**: Edge device specifications including sensor types, sampling rates, battery capacity, storage capacity, and processing capabilities
- **Trend Summary**: Aggregated statistics over time period (week/month), including average AHI, event count trends, and pattern insights

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete a full night's sleep monitoring (8 hours) on a single battery charge
- **SC-002**: System detects apnea events with ≥95% sensitivity compared to clinical polysomnography ground truth
- **SC-003**: False positive rate for apnea detection is ≤10% (specificity ≥90%)
- **SC-004**: Users can review previous night's sleep report within 30 seconds of waking
- **SC-005**: System operates reliably without internet connectivity for 90+ consecutive days

### ML Model Success Criteria

- **SC-ML-001**: Model achieves ≥90% accuracy on held-out validation dataset for apnea event detection
- **SC-ML-002**: Per-class precision ≥85%, recall ≥90%, F1 ≥87% for apnea vs normal breathing classification
- **SC-ML-003**: False negative rate for severe apnea events (>30s duration) ≤5%
- **SC-ML-004**: Inference latency on target edge device ≤100ms per 30-second analysis window
- **SC-ML-005**: Model accuracy degradation on edge vs training environment <2%
- **SC-ML-006**: Model size fits within 50MB memory constraint on target wearable device
- **SC-ML-007**: Event classification model (obstructive/central/mixed) achieves ≥90% accuracy vs clinical labels
- **SC-ML-008**: AHI score calculation accuracy within ±2 events/hour compared to clinical polysomnography

## Assumptions

- Target users have been advised by healthcare provider to monitor for sleep apnea
- Users can properly position and wear the sensing device during sleep
- Edge device includes appropriate sensors (e.g., respiratory effort bands, pulse oximeter, accelerometer)
- Device has sufficient processing power for real-time ML inference (e.g., ARM Cortex-M or better)
- Users will periodically sync data to companion app or computer for backup and detailed analysis
- Clinical validation will be performed against standard polysomnography before medical use
- Regulatory approval (FDA/CE marking) will be pursued if device is marketed for medical diagnosis
- System will include appropriate medical disclaimers that it assists detection but doesn't replace professional diagnosis

## Out of Scope

- Treatment recommendations or prescriptions (detection only, not treatment)
- Integration with CPAP machines or other treatment devices
- Direct diagnosis of sleep apnea (requires healthcare provider interpretation)
- Monitoring other sleep disorders beyond breathing-related events
- Real-time alerts during sleep (future enhancement, current version logs for post-sleep review)
- Multi-user profiles on single device
- Cloud-based model training or updates (edge inference only)
