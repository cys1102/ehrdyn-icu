-- Requires official MIMIC-IV derived concepts and 00_base_eligible_stays.sql.
CREATE MATERIALIZED VIEW ehrdyn_icu_internal.frozen_cohort_anchors AS
WITH sepsis AS (
    SELECT b.*, 'kdd2027_sepsis_vasopressor_3bin'::text AS task_id,
           s.suspected_infection_time AS anchor_time
    FROM ehrdyn_icu_internal.base_eligible_stays AS b
    JOIN mimiciv_derived.sepsis3 AS s USING (stay_id)
    WHERE s.sofa_score >= 2
),
nonsepsis_base AS (
    SELECT b.*
    FROM ehrdyn_icu_internal.base_eligible_stays AS b
    LEFT JOIN sepsis AS s USING (stay_id)
    WHERE s.stay_id IS NULL
),
medication_anchor AS (
    SELECT b.stay_id,
           MIN(i.starttime) FILTER (WHERE LOWER(d.label) ~ 'furosemide|bumetanide|torsemide') AS diuretic_anchor,
           MIN(i.starttime) FILTER (WHERE LOWER(d.label) ~ 'amiodarone|diltiazem|metoprolol|esmolol') AS rate_rhythm_anchor,
           MIN(i.starttime) FILTER (WHERE LOWER(d.label) ~ 'norepinephrine|epinephrine|vasopressin|phenylephrine') AS vasopressor_anchor,
           MIN(i.starttime) FILTER (WHERE LOWER(d.label) ~ 'norepinephrine|epinephrine|vasopressin|phenylephrine|dobutamine|milrinone') AS hemodynamic_anchor
    FROM nonsepsis_base AS b
    JOIN mimiciv_icu.inputevents AS i
      ON i.stay_id = b.stay_id AND i.starttime >= b.intime AND i.starttime < b.outtime
    JOIN mimiciv_icu.d_items AS d USING (itemid)
    GROUP BY b.stay_id
),
respiratory AS (
    SELECT b.*, 'kdd2027_respiratory_peep_5bin'::text AS task_id,
           MIN(v.starttime) AS anchor_time
    FROM nonsepsis_base AS b
    JOIN mimiciv_derived.ventilation AS v USING (stay_id)
    GROUP BY b.subject_id, b.hadm_id, b.stay_id, b.intime, b.outtime, b.mortality_90d
),
aki AS (
    SELECT b.*, 'kdd2027_aki_diuretic_rrt_factorized_3bin'::text AS task_id,
           MIN(k.charttime) AS anchor_time
    FROM nonsepsis_base AS b
    JOIN mimiciv_derived.kdigo_stages AS k USING (stay_id)
    WHERE k.aki_stage >= 1
    GROUP BY b.subject_id, b.hadm_id, b.stay_id, b.intime, b.outtime, b.mortality_90d
),
diagnoses AS (
    SELECT hadm_id,
           BOOL_OR((icd_version = 10 AND REPLACE(UPPER(icd_code), '.', '') LIKE 'I50%') OR (icd_version = 9 AND REPLACE(icd_code, '.', '') LIKE '428%')) AS heart_failure,
           BOOL_OR((icd_version = 10 AND REPLACE(UPPER(icd_code), '.', '') LIKE 'I48%') OR (icd_version = 9 AND REPLACE(icd_code, '.', '') IN ('42731', '42732'))) AS af_flutter,
           BOOL_OR((icd_version = 10 AND (REPLACE(UPPER(icd_code), '.', '') LIKE 'I21%' OR REPLACE(UPPER(icd_code), '.', '') LIKE 'I22%')) OR (icd_version = 9 AND REPLACE(icd_code, '.', '') LIKE '410%')) AS ami
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY hadm_id
),
hf AS (
    SELECT b.*, 'kdd2027_hf_diuretic_binary'::text AS task_id,
           COALESCE(m.diuretic_anchor, b.intime) AS anchor_time
    FROM nonsepsis_base AS b
    JOIN diagnoses AS d USING (hadm_id)
    LEFT JOIN medication_anchor AS m USING (stay_id)
    WHERE d.heart_failure
),
af AS (
    SELECT b.*, 'kdd2027_af_rate_anticoag_compact_3bin'::text AS task_id,
           COALESCE(m.rate_rhythm_anchor, b.intime) AS anchor_time
    FROM nonsepsis_base AS b
    JOIN diagnoses AS d USING (hadm_id)
    LEFT JOIN medication_anchor AS m USING (stay_id)
    WHERE d.af_flutter
),
ami AS (
    SELECT b.*, 'kdd2027_ami_hemodynamic_compact_3bin'::text AS task_id,
           COALESCE(m.hemodynamic_anchor, b.intime) AS anchor_time
    FROM nonsepsis_base AS b
    JOIN diagnoses AS d USING (hadm_id)
    LEFT JOIN medication_anchor AS m USING (stay_id)
    WHERE d.ami
),
low_map AS (
    SELECT b.stay_id, v.charttime
    FROM nonsepsis_base AS b
    JOIN mimiciv_derived.vitalsign AS v USING (stay_id)
    WHERE v.mbp < 65
),
sustained_hypotension AS (
    SELECT first_low.stay_id, MIN(first_low.charttime) AS anchor_time
    FROM low_map AS first_low
    WHERE EXISTS (
        SELECT 1 FROM low_map AS confirmatory
        WHERE confirmatory.stay_id = first_low.stay_id
          AND confirmatory.charttime > first_low.charttime
          AND confirmatory.charttime <= first_low.charttime + INTERVAL '4 hour'
    )
    GROUP BY first_low.stay_id
),
shock AS (
    SELECT b.*, 'kdd2027_shock_fluid_bolus_binary'::text AS task_id,
           LEAST(s.anchor_time, m.vasopressor_anchor) AS anchor_time
    FROM nonsepsis_base AS b
    LEFT JOIN sustained_hypotension AS s USING (stay_id)
    LEFT JOIN medication_anchor AS m USING (stay_id)
    WHERE s.anchor_time IS NOT NULL OR m.vasopressor_anchor IS NOT NULL
)
SELECT * FROM sepsis
UNION ALL SELECT * FROM respiratory
UNION ALL SELECT * FROM aki
UNION ALL SELECT * FROM hf
UNION ALL SELECT * FROM af
UNION ALL SELECT * FROM ami
UNION ALL SELECT * FROM shock;
