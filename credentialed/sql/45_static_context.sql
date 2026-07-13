-- Static features used by the 40-feature contract; keep output local.
CREATE MATERIALIZED VIEW ehrdyn_icu_internal.static_context AS
WITH admission_order AS (
    SELECT subject_id, hadm_id,
           ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS admission_number
    FROM mimiciv_hosp.admissions
),
diagnosis_proxy AS (
    SELECT hadm_id,
           COUNT(DISTINCT LEFT(REPLACE(UPPER(icd_code), '.', ''), 3)) FILTER (
             WHERE NOT (REPLACE(UPPER(icd_code), '.', '') LIKE '038%'
                     OR REPLACE(UPPER(icd_code), '.', '') LIKE 'A40%'
                     OR REPLACE(UPPER(icd_code), '.', '') LIKE 'A41%')
           ) AS elixhauser_score_proxy
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY hadm_id
)
SELECT a.subject_id, a.hadm_id, a.stay_id, a.task_id,
       p.anchor_age::double precision AS age,
       CASE WHEN UPPER(p.gender) = 'M' THEN 1.0 ELSE 0.0 END AS gender_male,
       CASE WHEN o.admission_number > 1 THEN 1.0 ELSE 0.0 END AS readmission,
       COALESCE(d.elixhauser_score_proxy, 0)::double precision AS elixhauser_score_proxy
FROM ehrdyn_icu_internal.frozen_cohort_anchors AS a
JOIN mimiciv_hosp.patients AS p USING (subject_id)
JOIN admission_order AS o USING (subject_id, hadm_id)
LEFT JOIN diagnosis_proxy AS d USING (hadm_id);
