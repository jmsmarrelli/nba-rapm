# Source data

`source_manifest.json` records the exact input files, public download URLs,
byte sizes and SHA-256 hashes used for the September 2026 corrected release.
Season numbers denote the year the NBA season ends.

From the repository folder, run `python tools/fetch_data.py` to download missing
inputs and verify all required hashes. Use `python tools/fetch_data.py --check`
to verify without downloading. Changed upstream bytes stop the build; review
and deliberately version a new data release rather than silently mixing sources.

The shot studies use DomSamangy's public NBA shot-chart mirror; schedule,
box-score and play-by-play studies use the hoopR-data mirror. These files
are third-party data: the repository's MIT code license does not relicense them.
Raw data and processed caches are excluded from the publication package.
