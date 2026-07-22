# Frozen KDD220M2L input-layout contract

The preflight accesses only the exact relative paths declared by the 14 public
`TableContract` entries. If one encoding exists it is exposed. If both exist,
plain CSV is compared with the decompressed CSV.GZ stream in bounded chunks.
Only equal duplicates may be represented, and the view selects CSV.GZ.

The view must be external to the repository, named `3.1`, read-only, and made
of symbolic links. It must expose exactly one encoding per table and satisfy
every frozen required-header check. No constructor, cohort logic, schema,
task contract, or scientific setting is changed by this stage.
