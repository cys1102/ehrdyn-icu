# KDD242 result audit

## Outcome

EHRDyn-ICU v1.3.0 is published as an immutable annotated tag and public GitHub
Release at commit `3175349e2e922e6b352ac3fa7674537a810560bd`.

The release contains entrant-owned point, independent-Gaussian, and Gaussian-
ensemble examples; typed schemas; recursive evaluation; uncertainty scoring;
frozen support-only H4 planning with full-episode evaluation; direct-return and
repeated-dataset OPE entry points; task/metric documentation; aggregate KDD235B
evidence; dependency locks; and checksum/privacy validators.

## Verification

- The annotated tag peels to the release commit exactly.
- The GitHub Release is public, non-draft, and non-prerelease.
- Fresh public clone: 133/133 tests and 11/11 schemas pass.
- KDD235B: 40 checkpoints, 440 horizon rows, 40 direct-return rows, and
  240 environment-estimator OPE summaries complete.
- Published archive SHA-256:
  `cbbdf99e5af7ff0bfeca320f723bcb09d413ee1e42c6acbf385ab5e645dd2ccd`.
- Published archive: 558 checksum entries and 559 privacy-scanned files pass.
- Published quickstart reproduces metrics hash `3644a183...9c30e` and policy
  probability hash `192948f2...0735`.
- No built-in transition model is imported by the demonstration entrant, and
  it is not present in the scientific leaderboard.

No DOI was minted because no already configured archival integration was used.

## Claim boundary

The release demonstrates constructed-benchmark interface usability. It does
not establish independent EHR reconstruction, disease-faithful simulation,
clinical utility, treatment effects, policy benefit, or deployment readiness.
