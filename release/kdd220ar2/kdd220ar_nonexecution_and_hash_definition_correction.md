# KDD220AR nonexecution and hash-definition correction

KDD220AR stopped at its mandatory preservation gate because its prompt called
the secondary per-file-manifest digest the repository-standard tree hash. It
created no branch, edit, test, commit, push, or MIMIC access.

KDD220AR2 corrects the specification without relabeling KDD220AR as complete:

- repository-standard identity: sorted POSIX relative path bytes followed by
  file bytes, yielding `db55da00…bacb4bd`;
- secondary manifest identity: SHA-256 of newline-terminated
  `relative_path|file_sha256` rows, yielding `d1a2e187…b229daa`.

Both identities bind the same preserved 16-file KDD220B tree.
