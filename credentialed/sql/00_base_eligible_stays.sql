-- Run only in a credentialed MIMIC-IV v3.1 PostgreSQL environment.
-- This internal view contains join keys and must never be exported publicly.
CREATE SCHEMA IF NOT EXISTS ehrdyn_icu_internal;

CREATE MATERIALIZED VIEW ehrdyn_icu_internal.base_eligible_stays AS
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    CASE
        WHEN p.dod IS NOT NULL AND p.dod <= i.intime + INTERVAL '90 day' THEN 1
        ELSE 0
    END AS mortality_90d
FROM mimiciv_icu.icustays AS i
JOIN mimiciv_hosp.patients AS p USING (subject_id)
WHERE p.anchor_age >= 18
  AND i.outtime >= i.intime + INTERVAL '24 hour';
