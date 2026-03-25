# RUSH Gait Analysis Pipeline — Overview

---

## What Does This Pipeline Do?

This pipeline takes **raw wrist accelerometer data** from a participant's recording device and automatically produces a rich set of **walking quality and quantity metrics** — without any manual annotation.

There are **two main outputs** shared per cohort:

### Output 1 — Bout-Level Table

One CSV file per participant; each row is a single detected walking bout. Columns:

| Column | Description |
|--------|-------------|
| `subject_id` | Participant identifier |
| `bout_id` | Sequential index of the bout within the recording |
| `start_time` | Timestamp of the bout's start |
| `duration_sec` | Bout duration in seconds |
| `day` | Day index within the recording (0 = first full day) |
| `n_windows` | Number of 10-second windows that make up the bout |
| `total_steps` | Total step count for the bout |
| `speed` | Median walking speed across windows (m/s) |
| `cadence` | Median cadence across windows (steps/min) |
| `gait_length` | Median stride length across windows (m), from the deep learning model |
| `gait_length_indirect` | Stride length estimated from speed and cadence: (120 × speed) / cadence (m) |
| `regularity_eldernet` | Median gait regularity across windows (0–1), from the deep learning model |
| `entropy` | Sample entropy of the bout's acceleration signal — higher = more unpredictable movement |
| `pa_amplitude` | Mean amplitude of the acceleration magnitude during the bout (g) |
| `pa_variability` | Standard deviation of the acceleration magnitude within the bout (g) |

### Output 2 — Summary Statistics Table

One CSV file for the full cohort; each row is one participant, with over 200 columns summarizing their entire walking profile across all bouts and days. These are described in detail in the [Summary Statistics](#summary-statistics) section below.

---

## Pipeline Stages

The pipeline runs in three main stages:

```
Raw accelerometer data
        |
        v
  1. GAIT DETECTION          — "When was this person walking?"
        |
        v
  2. GAIT QUALITY ESTIMATION — "How did they walk?"
        |
        v
  3. SUMMARY STATISTICS      — "What does their overall walking profile look like?"
```

---

## Stage 1 — Gait Detection

**Goal:** Find every moment in the recording when the participant was walking.

The raw accelerometer signal is split into short overlapping windows (10 seconds each). A deep learning model — **ElderNet**, trained specifically on older adults — classifies each window as *walking* or *not walking*. Windows that show no movement at all (the participant is sitting still or the device is not being worn) are filtered out before classification to save computation time.

Adjacent walking windows are then merged into continuous **walking bouts**: uninterrupted episodes of walking lasting at least 10 seconds. Short gaps between windows (up to 3 seconds) are bridged over, since a brief pause mid-walk should not be treated as two separate events.

After bout detection, basic outlier filters are applied: bouts with duration exceeding 10,000 seconds (~2.5 hours) or with mean acceleration magnitude (`pa_amplitude`) greater than 5 g are removed as implausible.

**Output of this stage:** A list of all detected walking bouts, each with a start time and duration.

---

## Stage 2 — Gait Quality Estimation

**Goal:** For each detected walking bout, characterize *how* the person walked.

Two complementary approaches are used:

### Deep learning features
The same ElderNet model that detects walking also estimates — window by window — four core gait metrics:

| Metric | What it measures | Units |
|--------|-----------------|-------|
| **Speed** | How fast the person walks | m/s |
| **Cadence** | How many steps per minute | steps/min |
| **Stride length** | How long each stride is | m |
| **Regularity** | How consistent stride-to-stride rhythm is | 0–1 scale |

These are averaged (median) across windows to give a single value per bout.

### Signal-processing features
Additional features are computed directly from the raw acceleration signal of each bout:

| Feature | What it captures |
|---------|-----------------|
| **Sample entropy** | Unpredictability / complexity of the movement signal |
| **Physical activity intensity (acceleration magnitude)** | Overall intensity of movement during the bout (g) |

**Output of this stage:** A **bout-level table** — one row per walking bout, per participant.

---

## Stage 3 — Summary Statistics

**Goal:** Condense hundreds or thousands of walking bouts into a single, rich participant-level profile.

A participant wearing a device for 7 days may accumulate **hundreds of walking bouts**. The summary statistics step compresses all of that information into a single row per participant, capturing not just the *average* walking pattern but also its *variability*, *day-to-day consistency*, and *distributional shape*.

**Output of this stage:** A **summary table** — one row per participant.

---

## Summary Statistics {#summary-statistics}

The statistics are organized into the categories below. For each category, the focus is on explaining metrics that go beyond straightforward averages and percentiles.

---

### Classical Summary Statistics

For each gait metric (speed, cadence, stride length, regularity, entropy, dominant frequency, bout duration, physical activity intensity, and others), the following summary statistics are computed across all bouts. The standard descriptors — median, mean, standard deviation, P10, P90 — are self-explanatory. The less familiar ones are described below.

| Statistic | Description |
|-----------|-------------|
| **IQR** (Interquartile Range) | The range covered by the middle 50% of bouts (P75 − P25). Robust to extreme values at either tail. |
| **CV** (Coefficient of Variation) | Standard deviation divided by the mean. Expresses spread as a fraction of the typical value, making it possible to compare variability across metrics with different scales (e.g., speed in m/s vs. cadence in steps/min). |
| **Skewness** | Asymmetry of the distribution. Positive skewness means a longer tail toward high values (e.g., occasional fast bouts in an otherwise slow walker); negative means a tail toward low values. |
| **Kurtosis** | Whether extreme values are unusually frequent. High kurtosis ("heavy tails") means the participant occasionally produces outlier bouts — very fast or very slow — more often than a normal distribution would predict. |
| **MAD** (Median Absolute Deviation) | The median of the absolute deviations from the median. A robust alternative to standard deviation that is not inflated by outliers. Computed only when at least 30 bouts are available. |
| **L-skewness** | A robust version of skewness derived from L-moments (linear combinations of order statistics). Less sensitive to outliers than conventional skewness, making it more reliable when the distribution has heavy tails. Requires ≥ 30 bouts. |
| **L-kurtosis** | A robust version of kurtosis, also from L-moments. Measures tail heaviness with less distortion from extreme values than conventional kurtosis. Requires ≥ 30 bouts. |

These statistics are computed per metric, producing columns such as `bout_speed_median`, `bout_speed_cv`, `bout_cadence_mad`, `bout_gait_length_l_skewness`, and so on.

---

### Distributional Shape Metrics

Applied to the four core gait metrics (speed, cadence, stride length, regularity), these statistics characterize the *shape* of a participant's overall distribution — beyond what mean and variance capture.

| Statistic | What it measures |
|-----------|-----------------|
| **Dip statistic** (`dist_*_dip_stat`) | Hartigan's Dip Test: quantifies how far the empirical distribution is from unimodal. A high dip statistic (with a low p-value) suggests the participant walks in two or more distinct modes — for example, a slow shuffling pace and a faster purposeful pace. |
| **Dip p-value** (`dist_*_dip_pvalue`) | The p-value associated with the dip test. Values below 0.05 indicate statistically significant multimodality. |
| **Shannon entropy** (`dist_*_shannon_entropy`) | Computed from the histogram of the metric. High entropy means bouts are spread relatively evenly across the speed (or cadence, etc.) range — the participant has no strongly preferred pace. Low entropy means most bouts cluster in one or two bins. |
| **Wasserstein distance to normal** (`dist_*_wasserstein_to_normal`) | The Earth Mover's Distance between the participant's actual distribution and a fitted Gaussian of the same mean and standard deviation. A high value means the distribution departs substantially from bell-curve shape — e.g., it is skewed, multimodal, or has heavy tails. Computed by fitting a Gaussian to the data and measuring how much "mass" must be moved to transform one distribution into the other. |

---

### Day-to-Day Stability

These metrics capture how consistent a participant's gait *distribution* is from one day to the next — not just whether their average speed changed, but whether the whole shape of the distribution shifted.

For each core gait metric, daily distributions are constructed from all bouts on each day (days with fewer than 30 bouts are excluded). Pairwise **Wasserstein distances** are then computed between every pair of days.

| Statistic | Description |
|-----------|-------------|
| `stability_*_mean_wasserstein` | Average of all pairwise day-to-day Wasserstein distances. Low = the participant walks similarly every day; high = substantial day-to-day variation in the gait distribution. |
| `stability_*_std_wasserstein` | Standard deviation of the pairwise distances. High = the amount of daily variation is itself inconsistent (some pairs of days are very different, others are not). |
| `stability_*_n_day_pairs` | Number of day-pairs used in the computation (depends on how many days had sufficient bouts). |

These are computed for speed, cadence, stride length, and regularity.

---

### Within-Bout Variability (Stride-to-Stride)

Standard bout-level metrics (e.g., `bout_speed_median`) describe each bout by a single number. Within-bout variability goes further: for each bout, the **standard deviation of the metric across its constituent windows** is computed. This captures how much gait fluctuates *within* a single walking episode.

The resulting per-bout standard deviations are then summarized across all bouts:

| Statistic | Description |
|-----------|-------------|
| `var_var_*_median` | Typical within-bout variability — the median of all per-bout standard deviations. Reflects how much stride-to-stride fluctuation is normal for this participant. |
| `var_var_*_std` | Spread of within-bout variability across bouts. High = the amount of stride-to-stride inconsistency is itself inconsistent — some bouts are very smooth, others erratic. |

Computed for speed, cadence, stride length, and regularity. Only bouts with at least 5 windows contribute.

---

### Daily Walking Volume & Bout Counts

How much walking the participant accumulates each day.

| Statistic | Description |
|-----------|-------------|
| `daily_walking_*` | Summary statistics (median, mean, std, p10, p90, IQR, CV) for **daily walking minutes** — the total duration of all bouts on each day. |
| `daily_step_count_*` | Summary statistics for **total steps per day**, aggregated across all bouts. |
| `daily_n_bouts_*` | Summary statistics for the **number of walking bouts per day**. |
| `daily_n_bouts_gt30s_*` | Summary statistics for the number of bouts **longer than 30 seconds** per day. Longer bouts may reflect more sustained, intentional walking. |
| `daily_n_bouts_gt60s_*` | Summary statistics for the number of bouts **longer than 60 seconds** per day. |
| `n_bouts` | Total walking bouts across the entire recording period. |
| `wear_days` | Number of valid recording days included in the analysis. |

---

### Time-of-Day Distribution

When during the day does walking occur? Bouts are categorized into four time windows:

| Statistic | Time window | Description |
|-----------|-------------|-------------|
| `tod_morning_proportion` / `_n_bouts` | 06:00–11:59 | Share and count of morning bouts |
| `tod_afternoon_proportion` / `_n_bouts` | 12:00–16:59 | Share and count of afternoon bouts |
| `tod_evening_proportion` / `_n_bouts` | 17:00–23:59 | Share and count of evening bouts |
| `tod_night_proportion` / `_n_bouts` | 00:00–05:59 | Share and count of night bouts |

Proportions sum to 1 across the four periods.

---

### Physical Activity Intensity

These metrics measure overall movement intensity — not just during detected walking bouts, but across the full day. Intensity is derived from the acceleration magnitude (the Euclidean norm of the tri-axial accelerometer signal), expressed in *g* (where g = gravitational acceleration ≈ 9.81 m/s²).

| Statistic | Description |
|-----------|-------------|
| `pa_amplitude_*` | Summary statistics for mean amplitude of the acceleration magnitude *during walking bouts* (g). Reflects how vigorously the participant walks. |
| `pa_variability_*` | Summary statistics for the variability of the acceleration magnitude within walking bouts (g). |
| `daily_pa_mean_*` | Summary statistics for mean acceleration magnitude across *the full day* (including non-walking periods). Reflects overall daily activity level. |
| `daily_pa_std_*` | Summary statistics for within-day variability of intensity — how much the activity level fluctuates throughout the day. |
| `tdpa_*` | Summary statistics for **Total Daily Physical Activity** — the integral of acceleration magnitude over the full day (g·s). A single-number summary of the day's total movement dose. |
