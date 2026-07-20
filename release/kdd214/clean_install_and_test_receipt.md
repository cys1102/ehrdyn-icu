# Clean install and test receipt

- Clean clone commit: `c42dd0d657efffda9023bc906c69ca6737055855`
- Installer: `uv sync --frozen` on Python 3.11.14; seven runtime packages installed
- Released schemas: 3/3 valid and bound
- Complete tests: 50 passed; 2 credentialed-extra tests skipped as declared
- Cross-Python focused tests: 17/17 passed on each of 3.11, 3.12, and 3.13
- Measured clean verification wall time: 3 seconds
- Clean-clone computed hash matched the frozen expected hash exactly
- Privacy and checksum scans passed
