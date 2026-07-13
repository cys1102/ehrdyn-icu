-- Long-format raw observation events for the public 40-feature contract.
-- The output is restricted and must remain inside the credentialed environment.
CREATE MATERIALIZED VIEW ehrdyn_icu_internal.observation_events AS
WITH chart_events AS (
    SELECT w.subject_id, w.stay_id, w.task_id, w.step_index,
           CASE c.itemid
             WHEN 220045 THEN 'heart_rate'
             WHEN 220050 THEN 'sbp' WHEN 220179 THEN 'sbp' WHEN 225309 THEN 'sbp'
             WHEN 220051 THEN 'dbp' WHEN 220180 THEN 'dbp' WHEN 225310 THEN 'dbp'
             WHEN 220052 THEN 'mbp' WHEN 220181 THEN 'mbp'
             WHEN 220210 THEN 'respiratory_rate' WHEN 224688 THEN 'respiratory_rate' WHEN 224689 THEN 'respiratory_rate' WHEN 224690 THEN 'respiratory_rate'
             WHEN 223761 THEN 'temperature_c' WHEN 223762 THEN 'temperature_c' WHEN 226329 THEN 'temperature_c' WHEN 228242 THEN 'temperature_c'
             WHEN 220277 THEN 'spo2'
             WHEN 220739 THEN 'gcs_proxy' WHEN 223900 THEN 'gcs_proxy' WHEN 223901 THEN 'gcs_proxy' WHEN 227013 THEN 'gcs_proxy'
             WHEN 223835 THEN 'fio2' WHEN 226754 THEN 'fio2' WHEN 227010 THEN 'fio2' WHEN 229280 THEN 'fio2'
             WHEN 224639 THEN 'weight' WHEN 226512 THEN 'weight' WHEN 226531 THEN 'weight'
             WHEN 225792 THEN 'mechanical_ventilation' WHEN 225794 THEN 'mechanical_ventilation'
           END AS feature_name,
           CASE
             WHEN c.itemid = 223761 THEN (c.valuenum - 32.0) * 5.0 / 9.0
             WHEN c.itemid = 226531 THEN c.valuenum * 0.45359237
             WHEN c.itemid IN (225792, 225794) THEN 1.0
             WHEN c.itemid IN (223835, 226754, 227010, 229280) AND c.valuenum BETWEEN 0 AND 1 THEN c.valuenum * 100.0
             ELSE c.valuenum
           END AS feature_value
    FROM ehrdyn_icu_internal.four_hour_windows AS w
    JOIN mimiciv_icu.chartevents AS c
      ON c.stay_id = w.stay_id AND c.charttime >= w.window_start AND c.charttime < w.window_end
    WHERE c.itemid IN (220045,220050,220179,225309,220051,220180,225310,220052,220181,220210,224688,224689,224690,223761,223762,226329,228242,220277,220739,223900,223901,227013,223835,226754,227010,229280,224639,226512,226531,225792,225794)
      AND c.valuenum IS NOT NULL
),
lab_events AS (
    SELECT w.subject_id, w.stay_id, w.task_id, w.step_index,
           CASE l.itemid
             WHEN 50813 THEN 'lactate' WHEN 52442 THEN 'lactate' WHEN 53154 THEN 'lactate'
             WHEN 50821 THEN 'pao2' WHEN 52042 THEN 'pao2'
             WHEN 50818 THEN 'paco2' WHEN 52040 THEN 'paco2'
             WHEN 50820 THEN 'ph'
             WHEN 50802 THEN 'base_excess' WHEN 52038 THEN 'base_excess'
             WHEN 50803 THEN 'co2_bicarbonate' WHEN 50882 THEN 'co2_bicarbonate' WHEN 52039 THEN 'co2_bicarbonate'
             WHEN 51300 THEN 'wbc' WHEN 51301 THEN 'wbc' WHEN 51755 THEN 'wbc' WHEN 51756 THEN 'wbc'
             WHEN 51265 THEN 'platelet' WHEN 53189 THEN 'platelet'
             WHEN 51006 THEN 'bun' WHEN 52647 THEN 'bun'
             WHEN 50912 THEN 'creatinine' WHEN 52024 THEN 'creatinine' WHEN 52546 THEN 'creatinine'
             WHEN 51275 THEN 'ptt' WHEN 52923 THEN 'ptt'
             WHEN 51274 THEN 'pt' WHEN 52921 THEN 'pt'
             WHEN 51237 THEN 'inr' WHEN 51675 THEN 'inr'
             WHEN 50878 THEN 'ast' WHEN 53088 THEN 'ast'
             WHEN 50861 THEN 'alt' WHEN 53084 THEN 'alt'
             WHEN 50885 THEN 'total_bilirubin' WHEN 53089 THEN 'total_bilirubin'
             WHEN 50960 THEN 'magnesium'
             WHEN 50808 THEN 'ionized_calcium' WHEN 51624 THEN 'ionized_calcium'
             WHEN 50893 THEN 'calcium' WHEN 52034 THEN 'calcium' WHEN 52035 THEN 'calcium'
           END AS feature_name,
           l.valuenum AS feature_value
    FROM ehrdyn_icu_internal.four_hour_windows AS w
    JOIN mimiciv_hosp.labevents AS l
      ON l.hadm_id = w.hadm_id AND l.charttime >= w.window_start AND l.charttime < w.window_end
    WHERE l.itemid IN (50813,52442,53154,50821,52042,50818,52040,50820,50802,52038,50803,50882,52039,51300,51301,51755,51756,51265,53189,51006,52647,50912,52024,52546,51275,52923,51274,52921,51237,51675,50878,53088,50861,53084,50885,53089,50960,50808,51624,50893,52034,52035)
      AND l.valuenum IS NOT NULL
),
urine_events AS (
    SELECT w.subject_id, w.stay_id, w.task_id, w.step_index,
           'urine_output'::text AS feature_name, SUM(o.value) AS feature_value
    FROM ehrdyn_icu_internal.four_hour_windows AS w
    JOIN mimiciv_icu.outputevents AS o
      ON o.stay_id = w.stay_id AND o.charttime >= w.window_start AND o.charttime < w.window_end
    WHERE o.itemid IN (226566,226627,226631,227489) AND o.value IS NOT NULL
    GROUP BY w.subject_id, w.stay_id, w.task_id, w.step_index
),
event_union AS (
    SELECT * FROM chart_events WHERE feature_name IS NOT NULL
    UNION ALL SELECT * FROM lab_events WHERE feature_name IS NOT NULL
    UNION ALL SELECT * FROM urine_events
)
SELECT subject_id, stay_id, task_id, step_index, feature_name,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY feature_value) AS feature_value
FROM event_union
GROUP BY subject_id, stay_id, task_id, step_index, feature_name;
