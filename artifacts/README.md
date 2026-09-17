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
The draw is deterministic, so re-running with the same seed and span reproduces
it byte for byte.
