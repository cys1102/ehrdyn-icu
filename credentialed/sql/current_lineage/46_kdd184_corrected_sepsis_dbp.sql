-- Additive KDD184 corrected sepsis DBP surface.
-- Preserve ehrdyn_icu_internal.observation_events as the historical v1.1.1 view.
-- Export this view in place of that historical view for the corrected sepsis lineage.
CREATE MATERIALIZED VIEW ehrdyn_icu_internal.observation_events_kdd184_corrected AS
WITH preserved AS (
    SELECT subject_id, stay_id, task_id, step_index, feature_name, feature_value
    FROM ehrdyn_icu_internal.observation_events
    WHERE NOT (task_id = 'kdd2027_sepsis_vasopressor_3bin' AND feature_name = 'dbp')
),
corrected_sepsis_dbp AS (
    SELECT w.subject_id, w.stay_id, w.task_id, w.step_index, 'dbp'::text AS feature_name,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY c.valuenum) AS feature_value
    FROM ehrdyn_icu_internal.four_hour_windows AS w
    JOIN mimiciv_icu.chartevents AS c
      ON c.stay_id = w.stay_id
     AND c.charttime >= w.window_start
     AND c.charttime < w.window_end
    WHERE w.task_id = 'kdd2027_sepsis_vasopressor_3bin'
      AND c.itemid IN (220051, 220180, 225310)
      AND c.valueuom = 'mmHg'
      AND c.valuenum IS NOT NULL
      AND c.valuenum BETWEEN 10.0 AND 200.0
    GROUP BY w.subject_id, w.stay_id, w.task_id, w.step_index
)
SELECT * FROM preserved
UNION ALL
SELECT * FROM corrected_sepsis_dbp;
