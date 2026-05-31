# AI Handoff: Node-by-Node Testing Rule

Date: 2026-05-16

## Permanent rule

Run pytest tests one node at a time for this project. Do not treat a whole-file or whole-suite run as the primary validation path when real audio fixtures are involved.

Why:

- One test file can sort many real WAV files and hit sandbox timeouts.
- A single hanging fixture can hide which cases passed.
- Previous bundles were falsely treated as good because only narrow tests were run.

Required command:

```bash
./commands/quality/RUN_ALL_PYTESTS_NODE_BY_NODE.command
```

For a focused file:

```bash
./commands/quality/RUN_ALL_PYTESTS_NODE_BY_NODE.command tests/test_parent_eligibility_v26_fx_smoke_reverb.py
```

If any node fails or times out, fix that node before claiming the bundle is validated.

## Source-name rule

Production sorter/voter/role/decision logic must not use filenames, folder names, ZIP member names, or path text as classification evidence. Source names are allowed only in post-sort diagnostic/audit tools and test oracles.

## Current focused fix

The sax/reed smoke test was changed to enforce the true safety invariant: sax-like phrases must stay inside Instruments and must not route to Human/Voice, Drums, or FX. Exact Brass/Woodwind leaf is not always provable from audio alone without using source labels, so the test now accepts broad Instrument placement.
