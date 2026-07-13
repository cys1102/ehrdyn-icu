-- Frozen action proxies. Validate local item labels against MIMIC-IV v3.1.
CREATE MATERIALIZED VIEW ehrdyn_icu_internal.action_exposures AS
WITH medication AS (
    SELECT w.stay_id, w.task_id, w.step_index,
           LOWER(COALESCE(d.label, '')) AS label,
           COALESCE(SUM(i.amount), 0.0) AS amount,
           COALESCE(MAX(COALESCE(i.rate, i.amount)), 0.0) AS max_rate
    FROM ehrdyn_icu_internal.four_hour_windows AS w
    JOIN mimiciv_icu.inputevents AS i
      ON i.stay_id = w.stay_id AND i.starttime < w.window_end AND COALESCE(i.endtime, i.starttime) >= w.window_start
    JOIN mimiciv_icu.d_items AS d USING (itemid)
    GROUP BY w.stay_id, w.task_id, w.step_index, LOWER(COALESCE(d.label, ''))
),
ventilator AS (
    SELECT w.stay_id, w.task_id, w.step_index,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY c.valuenum) AS peep
    FROM ehrdyn_icu_internal.four_hour_windows AS w
    JOIN mimiciv_icu.chartevents AS c
      ON c.stay_id = w.stay_id AND c.charttime >= w.window_start AND c.charttime < w.window_end
    WHERE c.itemid IN (220339, 224700) AND c.valuenum IS NOT NULL
    GROUP BY w.stay_id, w.task_id, w.step_index
),
renal_support AS (
    SELECT w.stay_id, w.task_id, w.step_index, 1 AS rrt_crrt
    FROM ehrdyn_icu_internal.four_hour_windows AS w
    JOIN mimiciv_icu.procedureevents AS p
      ON p.stay_id = w.stay_id AND p.starttime < w.window_end AND COALESCE(p.endtime, p.starttime) >= w.window_start
    JOIN mimiciv_icu.d_items AS d USING (itemid)
    WHERE LOWER(d.label) ~ 'dialysis|crrt|cvvh|renal replacement'
    GROUP BY w.stay_id, w.task_id, w.step_index
)
SELECT w.subject_id, w.stay_id, w.task_id, w.step_index, w.mortality_90d,
       COALESCE(SUM(m.amount) FILTER (WHERE m.label ~ 'saline|lactated ringer|fluid bolus'), 0.0) AS fluid_bolus,
       COALESCE(MAX(m.max_rate) FILTER (WHERE m.label ~ 'norepinephrine|epinephrine|vasopressin|phenylephrine'), 0.0) AS vasopressor,
       COALESCE(SUM(m.amount) FILTER (WHERE m.label ~ 'furosemide|bumetanide|torsemide'), 0.0) AS diuretic,
       COALESCE(MAX(m.max_rate) FILTER (WHERE m.label ~ 'dobutamine|milrinone'), 0.0) AS inotrope,
       COALESCE(SUM(m.amount) FILTER (WHERE m.label ~ 'amiodarone|diltiazem|metoprolol|esmolol'), 0.0) AS rate_rhythm_control,
       COALESCE(SUM(m.amount) FILTER (WHERE m.label ~ 'heparin|enoxaparin|warfarin'), 0.0) AS anticoagulation,
       COALESCE(r.rrt_crrt, 0) AS rrt_crrt,
       v.peep
FROM ehrdyn_icu_internal.four_hour_windows AS w
LEFT JOIN medication AS m USING (stay_id, task_id, step_index)
LEFT JOIN ventilator AS v USING (stay_id, task_id, step_index)
LEFT JOIN renal_support AS r USING (stay_id, task_id, step_index)
GROUP BY w.subject_id, w.stay_id, w.task_id, w.step_index, w.mortality_90d, v.peep, r.rrt_crrt;
