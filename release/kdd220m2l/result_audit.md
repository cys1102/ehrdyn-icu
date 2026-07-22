# KDD220M2L input-layout preflight audit

KDD220M2L inspected only the 14 table paths frozen in the public table
contract. It did not invoke the constructor and did not recursively discover
clinical files.

All 14 paths exposed both CSV and CSV.GZ encodings. For each path, the plain
CSV byte stream was compared incrementally with the decompressed CSV.GZ stream;
all 14 comparisons were equal. Required header fields passed for all 14
tables. A temporary external view rooted at a directory named `3.1` exposes
only the CSV.GZ encoding through read-only symbolic links. No clinical file
was copied or modified.

The private source and view paths, file contents, rows, identifiers, and
patient-level hashes are not recorded. Decision:
`complete_single_encoding_view_ready_for_kdd220m2s2`.
