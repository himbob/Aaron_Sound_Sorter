# Neural Audio Migration Phases

## Phase 0 — Contracts and shadow infrastructure (implemented)

Deliverables:

- source-blind provider interface;
- hash-keyed atomic embedding cache;
- disjoint `review_preview` and `heldout_eval` manifests;
- deterministic multi-prototype index;
- exact NumPy cosine classifier;
- prototype-radius OOD evidence;
- per-label encoder leaderboard and evidence-gated fusion decision;
- review-feedback log and logistic calibration seam;
- legacy-versus-neural disagreement CSV;
- leave-one-out trainer-contamination audit;
- optional CLAP and generic Hugging Face providers;
- no production routing changes.

Exit gate:

- all new architecture tests pass;
- no-source-name audit passes;
- the same audio under misleading filenames produces identical neural evidence.

Status: implemented. Focused contract tests and the project source-name audit
must remain green in every later phase.

## Phase 1 — CLAP vocal/instrument/drum shadow trial

1. Pin a reviewed CLAP snapshot locally.
2. Audit the existing correction corpus and remove or quarantine contaminated trainers.
3. Build curated training and held-out sets without byte reuse.
4. Start with the uploaded deep-house vocal pack plus negative controls:
   - sax and woodwinds;
   - synth leads/pads;
   - texture beds;
   - spoken/processed voice;
   - drum and percussion loops.
5. Build CLAP prototypes from preview examples only.
6. Evaluate held-out examples and produce category-level metrics.
7. Run the same inputs through legacy and neural systems.
8. Human-review disagreements first.

Exit gate:

- held-out vocal-family recall and false-positive rate are materially better than legacy;
- no regression panel category is made worse without an explicit documented tradeoff;
- OOD behavior is understandable and repeatable.

Status: partially executed on 2026-07-24 UTC. CLAP recall is 30/39 versus
legacy 21/39. Raw protected-negative voice false positives are 2/28, but both
are OOD; in-distribution false positives are 0/28. The vocal OOD/Review rate is
38/39. Clean breath/mouth/formant coverage is missing. The exit gate is not met.

## Phase 2 — Representation competition

1. Add one music/timbre-focused comparison lane:
   - MERT only if licensing and remote-code packaging are acceptable; otherwise
   - Essentia MusiCNN/MAEST/OpenL3 or BEATs.
2. Evaluate each provider independently per label.
3. Write a per-label leaderboard.
4. Choose one winner per label.
5. Test fusion only where it beats the winner on held-out data.

Exit gate:

- every active lane has unique measured value;
- no lane is retained merely because it is old or sounds sophisticated.

Status: executed as a research comparison on 2026-07-24. MERT-v1-95M layer 6
beat CLAP on drums, other instruments, and sax/woodwind, while CLAP beat MERT on
synth/pad/keys and musical vocal. MERT's checkpoint is CC-BY-NC-4.0, so the
registry keeps it research-only and records CLAP as the distributable candidate.
Fusion is disabled because no fused held-out result exists. The exit gate is
not met because sparse categories and the MERT license prevent deployment.

## Phase 3 — Calibrated shadow placement

1. Collect accepted/rejected disagreement decisions.
2. Train an inspectable logistic trust calibrator.
3. Include similarity, margin, radius ratio, prototype count, label support, and
   objective structure agreement.
4. Keep isotonic calibration disabled until there is enough data to avoid overfitting.
5. Produce predicted trust probability but still do not move files automatically.

Exit gate:

- probability calibration is measured on a later held-out time slice;
- review rate drops without increasing catastrophic cross-family errors.

## Phase 4 — Limited neural ownership

1. Allow neural ownership only for categories that pass their held-out gate.
2. Keep the legacy sorter running in shadow for comparison.
3. Keep a small safety boundary for invalid audio and objective structure conflicts.
4. Remove category-specific logic from the arbiter only after the neural owner replaces it.

Exit gate:

- rollback is one configuration switch;
- protected real-audio smoke cases pass;
- neural-owned categories show sustained superiority.

Status: started on 2026-07-24. Approved GUI corrections now enter a durable
source-blind inbox and rebuild versioned CLAP prototypes. Known neighborhoods
may own GUI proposals when measured structure agrees. Unknown cross-family
conflicts go to Review. Broader category ownership still requires held-out
coverage and calibration.

## Phase 5 — Arbiter retirement and brain reduction

1. Measure unique wins, unique vetoes, duplication, runtime, and review impact for every
   legacy brain lane.
2. Retire only lanes with no unique held-out value.
3. Extract remaining objective safeguards into small typed policies.
4. Delete or archive category rule branches only after replay proves replacement.
5. Reduce `FamilyClaimArbiter` to a small compatibility coordinator, then remove it.

Status: one coarse actual replay is complete. On 45 explicit locked labels, the
baseline and `--disable-baby-brain-ensemble` configurations both placed 45/45
exactly with no Review or final differences; disabling the group reduced the
classification stage from 51.747s to 42.022s in that run. This is not enough to
retire anything: independent core/spread/outlier toggles, safety-veto evidence,
and broader held-out panels are still missing.

## Phase 6 — Optional full-track capabilities

Only after the sample sorter is stable:

- stem separation for mixed tracks;
- audio-LLM review descriptions;
- approximate vector indexes for very large libraries;
- selective foundation-model fine-tuning.
