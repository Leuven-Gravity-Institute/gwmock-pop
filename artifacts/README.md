# Artifacts

Produced data that a downstream analysis consumes, kept out of the package wheel
(only `src/gwmock_pop` is packaged) so an installation stays light. These files
are small and are committed so a reference draw is available without a catalogue
download; the catalogues themselves are never committed.

## Files

- `catalogue-draw-reference.h5` — one seeded draw from the design-comparison
  catalogues at catalogue distances with the detector-frame band cut. The draw's
  provenance record is embedded in the file's `metadata` group: the pinned
  catalogue digests, the seed, the span and the draw settings.
- `catalogue-band-composition.json` — the composition stamp beside that draw:
  the per-class band composition over ten draws, the draw-to-draw spread, the
  full-catalogue anchor, and the draw file's own digest.

## Reproducibility

**The draw data are deterministic; the file bytes are not.** Re-running the
command below with the same seed, span and catalogues reproduces all 695 rows
and all 12 parameter columns exactly, and reproduces the composition numbers.
The file digest changes on every regeneration, because the embedded provenance
record is a record of the run and carries facts that legitimately differ between
runs: `created_utc`, and the commit of the checkout that produced the file. The
stamp is rewritten in the same run, so its `draw.sha256` always matches the file
committed beside it. This is why the claim here is data-level determinism rather
than byte-for-byte reproducibility: a reproducible timestamp would mean writing
a time the run did not happen at, and the producing commit is a fact about the
code, not a knob.

The per-row class labels are not stored in the HDF5. The catalogue conventions
accept canonical numeric parameter columns, and a non-numeric class column would
make the draw unreadable by `FilePopulationLoader` and unwritable as CSV, so the
labels live outside the column set. They are recoverable from the archived pair
and the draw: the provenance records the band edge and the intermediate-mass cut
(source-frame total mass ≥ 100 M☉), so in-band and intermediate-mass membership
are recomputable per row; the two source classes come from the two pinned
catalogues, whose source-frame primary-mass ranges are disjoint, so
binary-neutron-star versus binary-black-hole membership follows from the
source-frame primary mass; and re-running the deterministic draw returns the
labels exactly as `draw.population_class`. The stamp carries the per-class
aggregate counts.

## Regenerating

```bash
uv run gwmock-pop catalogue \
  --seed 42 --draws 10 --span-seconds 25600 \
  --output artifacts/catalogue-draw-reference.h5 \
  --stamp artifacts/catalogue-band-composition.json \
  --overwrite
```

The catalogues are fetched on demand into the package cache and verified against
their pinned SHA-256; add `--offline` to use a cache that is already populated.
Because the digest changes with the provenance record, regenerate both files
together so the stamp's `draw.sha256` describes the file actually committed.
