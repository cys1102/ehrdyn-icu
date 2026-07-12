# MIMIC-IV Access and Local Setup

MIMIC-IV v3.1 is credentialed data. This release does not redistribute it.

1. Create a PhysioNet account and complete credentialing.
2. Complete approved human-subject and data-privacy training. PhysioNet
   currently documents the CITI `Data or Specimens Only Research` route.
3. Sign the PhysioNet Credentialed Health Data Use Agreement for MIMIC-IV v3.1.
4. Obtain MIMIC-IV v3.1 through PhysioNet or an approved linked service.
5. Build the MIMIC-IV concepts required by the versioned task cards and local
   extraction implementation, including sepsis, SOFA, ventilation, vitals,
   vasopressor, and KDIGO concepts.
6. Run extraction only in the credentialed environment. Replace schema names
   through database configuration rather than copying source tables.

Official references:

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- Credentialed DUA: https://physionet.org/content/mimiciv/view-dua/3.1/
- Training instructions: https://physionet.org/about/citi-course/
- Credentialing FAQ: https://physionet.org/about/faqs/

Do not upload MIMIC-IV rows or restricted derivatives to public repositories,
third-party APIs, shared prompts, or unapproved online services. Keep local
database keys and split assignments inside the approved environment.
