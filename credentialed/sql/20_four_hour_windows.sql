-- Generates the fixed 18-step [-24h,+48h] grid; keep this view internal.
CREATE MATERIALIZED VIEW ehrdyn_icu_internal.four_hour_windows AS
SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.task_id,
    c.mortality_90d,
    step_index,
    c.anchor_time - INTERVAL '24 hour' + step_index * INTERVAL '4 hour' AS window_start,
    c.anchor_time - INTERVAL '24 hour' + (step_index + 1) * INTERVAL '4 hour' AS window_end
FROM ehrdyn_icu_internal.frozen_cohort_anchors AS c
CROSS JOIN generate_series(0, 17) AS step_index
WHERE c.anchor_time + INTERVAL '48 hour' > c.intime
  AND c.anchor_time - INTERVAL '24 hour' < c.outtime;
