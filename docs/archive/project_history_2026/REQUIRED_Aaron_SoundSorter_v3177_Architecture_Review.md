# Archived: Aaron Sound Sorter v31.77 — Static Architecture Review
## Family Claim Decision Ownership

**Reviewed:** `decision_core_v2.py` (2279 lines), `consensus.py` (1291 lines), `family_claims.py` (82 lines)  
**Not in bundle:** `eligibility.py`, `roles.py`, `role_gate.py`, `physics_voter.py`, `brain_voter.py`

> The five missing files are referenced heavily through imported types and function calls
> (`infer_parent_eligibility`, `EligibilityDecision`, `role_strength`, `detected_parent_role_name`,
> `compatible_tops_for_role`). The observations below are grounded in the three files present.
> Where eligibility behaviour is inferred it is marked as such.

---

## 1. Executive Summary

The expected architecture assigns final family placement to **one** arbiter.  
The current code has **two** modules that both issue final `ConsensusDecision` objects with real
folder paths, and each of those modules contains multiple internal escalation chains that each
issue their own decision. The count:

| Module | Method(s) emitting final `ConsensusDecision` | Call-sites that reach those methods |
|---|---|---|
| `consensus.py` — `ConsensusRunner` | 8 methods | ~14 early-return gates in `choose()` |
| `decision_core_v2.py` — `DecisionCoreV2` | 4 terminal methods (`_redirect_from_raw`, `_broaden_from_raw`, `_review_from_raw`, `_review_from_raw_with_role`) | **63** call-sites |

The 63 call-sites in `DecisionCoreV2` each embed a hardcoded folder path string (26 unique
paths are scattered across the file). That means folder-path logic that belongs in
`PlacementResolver` is duplicated across 26 locations in a single file.

`family_claims.py` is the only file that is architecturally clean: `FamilyClaim` is a
typed evidence object with no `folder_path` field, and `build_human_voice_claim` correctly
returns a claim rather than a decision. The problem is that `FamilyClaim` is only used in
one place inside `_early_candidate_adjudication`; the rest of the system ignores it and
builds decisions directly.

---

## 2. Every Function That Can Currently Override Final Folder/Family

### 2a. `consensus.py` — `ConsensusRunner`

All of these are called from `choose()` with early-return semantics. Whichever one fires
first wins; none of them consult each other.

| Method | What it decides | How it decides |
|---|---|---|
| `top_family_sanity_decision()` | Picks a new winner from `shared` candidates when the raw winner's `top_family` does not match the `dynamic_role_gate` | Rewrites `ConsensusDecision.folder_path` from the rescued candidate |
| `concrete_fx_gate_override()` | Protects concrete FX from a generic Instrument-loop gate | Rewrites `ConsensusDecision.folder_path` from the best concrete FX candidate |
| `shape_sanity_decision()` | Redirects to a shape-compatible family when raw winner is shape-incompatible | Rewrites `ConsensusDecision.folder_path` from the best compatible shared candidate |
| `role_sanity_decision()` | Enforces measured parent role; routes to compatible family or broad bucket | Has 6 sub-branches, each emitting `ConsensusDecision` or delegating to `broad_measured_role_decision()` |
| `broad_measured_role_decision()` | Synthesises a broad bucket from `synthetic_broad_bucket()` | Constructs `ConsensusDecision` from a **synthetic** row (not from any real voter candidate) |
| `same_family_role_bucket_decision()` | Broadens within the same family for bass/voice roles | Returns `ConsensusDecision` directly or via `broad_measured_role_decision()` |
| `prefer_profile_leaf_with_broad_support()` | Promotes a BrainVoter bass leaf when PhysicsVoter only supported the family | Constructs `ConsensusDecision` from a `CategoryGuess` — not a shared candidate |
| `choose()` itself (lines 348, 169) | Emits the "normal path" strong-consensus decision | Constructs `ConsensusDecision` from the winning shared-candidate row |

**Architecture violation:** `role_sanity_decision()`, `shape_sanity_decision()`, and
`top_family_sanity_decision()` each fully own final placement for the cases they handle.
They do not produce evidence; they produce the answer. That is arbiter behaviour inside a
runner that was supposed to produce a raw candidate set.

### 2b. `decision_core_v2.py` — `DecisionCoreV2`

The four terminal constructors are wrappers around `ConsensusDecision(...)` with hardcoded or
eligibility-derived paths. Every rescue branch in this file reduces to one of these four:

| Terminal method | What it constructs |
|---|---|
| `_redirect_from_raw(raw, folder_path, reason)` | A new `ConsensusDecision` with an explicit hardcoded `folder_path` string |
| `_broaden_from_raw(raw, eligibility, prefix)` | A `ConsensusDecision` whose `folder_path` comes from `eligibility.broad_folder_path` |
| `_review_from_raw(raw, eligibility, reason)` | A review `ConsensusDecision` (`_TO_REVIEW/Measured Role Conflict`) |
| `_review_from_raw_with_role(raw, role_name, reason)` | Same as above without an eligibility object |

The 26 hardcoded folder-path strings passed to `_redirect_from_raw()`:

```
"Drums/Drum Loops/Loops"                        ×8
"FX/Human and Voice FX"                         ×7
"Instruments/Instrument Loops/Loops"            ×4
"Instruments/Bass/Bass Loops"                   ×2
"Instruments/Brass and Woodwinds/Loops"         ×2
"Drums/Kick Drums/Generic Kick/One Shots"       ×1
"Drums/Percussion/Generic Percussion/One Shots" ×1
(plus safe_path variables that resolve to the above)
```

These strings are `PlacementResolver` output embedded directly in decision logic. Every time
a folder is renamed or a new sub-family is added, all 26 sites must be audited.

---

## 3. Every Branch That Converts a Role Directly Into a Folder

Each item below takes a measured or inferred role name and produces a folder path without
going through a claim-strength comparison.

### In `decision_core_v2.py`

**`_measured_broad_bucket_for_smoke_stability()` (~line 115)**  
- `role in pitched_roles` OR `shape in pitched_shapes and conf >= 0.72` → `"Instruments/Instrument Loops/Loops"` (unconditional redirect, no competing-claim check)  
- `role in drum_roles` OR `shape in drum_shapes and conf >= 0.70` → `"Drums/Drum Loops/Loops"` (broad path from eligibility, no competing-claim check)  
- `shape == "vocal_phrase" and conf >= 0.82` → `"FX/Human and Voice FX"` (see §4)  
- Final block: `eligibility.confidence >= 0.76 and not is_path_allowed` → `_broaden_from_raw()` (eligibility folder wins by confidence alone)

**`_true_bucket_for_candidate_conflict()` (~line 319)**  
- `eligibility.role_name == "fx_tonal_alert_or_siren"` + best instrument candidate → instrument path  
- `role == "pitched_percussion_loop"` → `_broaden_from_raw()` (eligibility folder, role drives it)  
- `role == "drum_loop" and low_rhythmic_drum_strength >= 0.75` → `"Drums/Drum Loops/Loops"` (role alone, no claim-strength check against voice)  

**`_early_candidate_adjudication()` (~line 613)**  
- `early_drum_loop_strength >= 0.55` + close drum candidate → `"Drums/Drum Loops/Loops"` (runs *before* all other checks)  
- `role == "bass_loop"` in eligibility + bass candidate → `"Instruments/Bass/Bass Loops"`  
- `role in {voiced_one_shot, vocal_music_phrase}` + `positive_voice_claim` → `"FX/Human and Voice FX"`  

**`_review_for_role_candidate_conflict()` (~line 1424)**  
- `role in {bass_loop, pitched_music_loop, mixed_music_loop}` + close kick candidate → review  
- `role in {pitched_reed_or_instrument_loop, pitched_reed_or_instrument_phrase}` + Sax eligibility path → `_broaden_from_raw()`  
(These are gating on role names to choose outcomes — better than the above because they go
to review rather than asserting a leaf, but still role-to-outcome logic outside the arbiter.)

### In `consensus.py`

**`role_sanity_decision()` (~line 771)**  
- `parent_role == "bass_loop" and low_rhythmic_strength >= 0.75 and not has_direct_brain_bass_support` → `broad_measured_role_decision("Drums", "low_rhythmic_drum_loop", …)` (role forces Drums family)  
- `parent_role == "bass_loop" and parent_strength >= 0.90 and has_brain_bass_support` → `broad_measured_role_decision("Instruments", "bass_loop", …)` (role forces Instruments family)  
- `parent_role in {bright_drum_loop, percussive_drum_loop, low_rhythmic_drum_loop}` → Drums family preferred  

**`same_family_role_bucket_decision()` (~line 974)**  
- `parent_role == "bass_loop" and parent_strength >= 0.90` → bass broad bucket  
- `parent_role in {voiced_one_shot, vocal_music_phrase}` → broad voice bucket (see §4)  

---

## 4. Every Branch That Treats Human/Voice as a Broad Rescue Bucket

These are the most common source of the voice-steals-sax/keys/bass regression.
They are listed in execution order within `DecisionCoreV2.apply_eligibility()`.

### `_measured_broad_bucket_for_smoke_stability()` (~line 290)
```python
if shape == "vocal_phrase" and shape_conf >= 0.82:
    if measured_role not in {pitched_music_loop, pitched_music_phrase, ...}:
        if shape_conf >= 0.95 or measured_role in voice_roles or role in voice_roles:
            return self._redirect_from_raw(raw, "FX/Human and Voice FX", ...)
```
**Risk:** Sax, flute, and some synths can produce `vocal_phrase` shape with high confidence
when they have breathy or formant-like envelopes. The only guard is checking `measured_role`;
if the measured role is ambiguous (e.g. `mixed_music_loop`), the shape gate fires.

### `_early_candidate_adjudication()` — six distinct Human/Voice paths (~lines 716–941)

**Path 1 (~line 716):** `direct_voice_strength >= 0.70` + raw is Drums/percussion → `"FX/Human and Voice FX"`.  
No `positive_voice_claim` check. Any instrument with a 0.70 `voiced_one_shot` score in the
direct/body view (piano attack transients, some synth stabs) can trigger this.

**Path 2 (~line 771):** Short-hit guard with voice candidate + `short_hit_voice_strength >= 0.70` →
`"FX/Human and Voice FX"`. The voice score threshold of 0.70 is loose enough to catch pitched
one-shots with formant character.

**Path 3 (~line 876):** `short_single_event and raw is FX/human-and-voice and not positive_voice_claim` → review (this is correct; it correctly refuses).

**Path 4 (~line 887):** `role in vocal_roles and positive_voice_claim` → `"FX/Human and Voice FX"`.  
Nominally requires `positive_voice_claim`, but `positive_voice_claim` is built from
`human_voice_claim.can_override` which is assembled from loose thresholds (see §7).

**Path 5 (~line 914):** `decisive_voice_candidate = positive_voice_claim and voice_beats_raw and voice_beats_drum and voice_beats_nonvoice` → `"FX/Human and Voice FX"` when raw is Instruments, Drums, or FX.  
This is the most legitimate path, but it fires for Instruments and FX together — FX includes
sax-like siren false-positives that already had their FX position for a reason.

**Path 6 (~line 926):** `max(measured_voice, direct_voice) >= 0.72` + `raw_is_drum_or_generic` →
`"FX/Human and Voice FX"` when any of: `positive_voice_claim`, `direct_one_shot_voice_claim`,
`strong_measured_vocal_shape`, `strong_direct_vocal_hit`.  
This is a compound OR condition. `strong_direct_vocal_hit` requires only `direct_voice >= 0.90`
and a hit shape — no competing instrument-claim check.

### `same_family_role_bucket_decision()` in `consensus.py` (~line 1058)
```python
formant_hit_voice_claim = bool(
    parent_role == "voiced_one_shot"
    and primary_shape == "hit_with_tail"
    and parent_strength >= 0.84
    and winner_role_fit >= 0.55      # ← threshold is too low
)
```
`winner_role_fit >= 0.55` on `voiced_one_shot` will pass for keys, piano, and some synth
one-shots whose transient+body spectral profile overlaps with vowel formants. When this fires,
it calls `broad_measured_role_decision()` which synthesises a Human/Voice bucket from a
**synthetic** row, not from any actual Human/Voice candidate in the voter lists.

### `_broad_voice_bucket_for_non_voice_instrument()` (~line 1811)
```python
if eligibility.role_name in {"vocal_phrase", "vocal_one_shot"}:
    if raw.final_top not in {"Instruments", "FX"}:
        return None
    if eligibility.confidence < 0.72:
        return None
    return self._broaden_from_raw(raw, eligibility, ...)
```
Fires whenever eligibility says `vocal_phrase/vocal_one_shot` and the raw winner is in
Instruments or FX — **without checking whether any actual Human/Voice candidate exists**.
Eligibility is inferred from measured audio only; this path can fire on reed instruments
that measure as vocal-phrase in their eligibility layer.

### `_broad_voice_bucket_when_voice_candidate_dominates_generic_loop()` (~line 1838)
This one is relatively safe (it requires voice candidate to beat raw rank by 8 points), but
it synthesises a new `EligibilityDecision` with `broad_folder_path="FX/Human and Voice FX"`
inline rather than delegating to FamilyClaimArbiter. That is still an ownership violation.

---

## 5. Every Branch That Treats Bass as a Fallback for Low Rhythmic Audio

### `_early_candidate_adjudication()` — bass_rescue_role block (~line 384)
```python
bass_rescue_role = (
    eligibility.role_name == "bass_loop"
    or measured_role == "bass_loop"        # ← "or measured_role" makes this very broad
    or (shape == "bass_phrase" and shape_conf >= 0.86 and eligibility.role_name == "bass_loop")
)
```
The `or measured_role == "bass_loop"` arm fires regardless of eligibility confidence. A drum
loop with a heavy 808-kick body can measure as `bass_loop` in the full-file view while
the direct/body view correctly identifies it as `low_rhythmic_drum_loop`. When
`measured_role == "bass_loop"` fires the rescue, the confidence check (`eligibility.confidence >= 0.78`)
may not be met by the drum-loop eligibility, but `measured_role` is checked **before** that
guard.

### `role_sanity_decision()` in `consensus.py` — bass_loop rescues (~lines 837–910)
Two separate bass rescues run sequentially:

**Rescue A (~line 855):** `parent_role == "bass_loop" and low_rhythmic_strength >= 0.75 and not has_direct_brain_bass_support` → forces **Drums/Drum Loops**.  
`has_direct_brain_bass_support` only checks `brain_result.guesses` for rank ≤ 6 with bass path
fragments. If BrainVoter ranked a bass leaf at rank 7, it won't block this, and the drum-loop
family wins. This is the correct direction but is fragile.

**Rescue B (~line 876):** `parent_role == "bass_loop" and parent_strength >= 0.90 and has_brain_bass_support` → forces **Instruments/Bass broad bucket**.  
The sequential ordering means Rescue A fires first. If `low_rhythmic_strength < 0.75`, Rescue A
is skipped and Rescue B fires without a drum-loop competing-claim check.

### `same_family_role_bucket_decision()` in `consensus.py` — bass_loop path (~line 998)
```python
if parent_role == "bass_loop" and parent_strength >= 0.90 and winner.top_family == "Instruments":
    return self.broad_measured_role_decision("Instruments", parent_role, ...)
```
This synthesises a bass bucket from a synthetic row regardless of whether any real Bass
candidate exists. A drum loop that measured as strong `bass_loop` in the full-file view
can land here if the raw winner was in Instruments.

---

## 6. Every Branch That Can Turn Drum Loops Into Bass One-Shots or Instrument Loops

### Short-hit guard in `_early_candidate_adjudication()` (~line 726)
```python
if (
    shape in {"bass_phrase", "hit_with_tail"}
    and shape_conf >= 0.72
    and not _is_measured_loop_phrase_context(shape, role, measured_role)
):
    raw_is_suspicious_one_shot = (
        ...
        or (raw.final_top == "Instruments" and _path_has_any(raw_path, ("instrument loops", "mixed musical loops")))
    )
```
When `role_sanity_decision()` in `ConsensusRunner` has already broadened a drum loop to a
generic `Instruments/Instrument Loops` bucket (because no Drums shared candidate survived the
combined-rank cut), the raw winner enters `DecisionCoreV2` with `final_top == "Instruments"`
and a path containing `"instrument loops"`. The short-hit guard then treats that as a
suspicious one-shot, looks for a drum candidate, and may route it to
`"Drums/Kick Drums/Generic Kick/One Shots"` — converting a loop to a one-shot.  
The guard against this (`_is_measured_loop_phrase_context`) only fires when `role` or
`measured_role` is in `LOOP_PHRASE_ROLES`. If the eligibility role is, say, `bass_loop`
(common for sub-heavy drum loops), that check fails and the short-hit gate opens.

### `role_sanity_decision()` in `consensus.py` — the drum-loop vs bass-loop contest (~line 837)
When `parent_role == "bass_loop"` and the raw winner is `Drums/Drum Loops`, the guard is:
```python
if drum_loop_role_strength >= 0.25 or strong_drum_loop_candidate:
    return None
```
`drum_loop_role_strength >= 0.25` is intentionally lenient (any small drum-loop signal
blocks the bass override). But `strong_drum_loop_candidate` requires `combined_rank_score <= 4.0`
which is a very tight threshold. If both voters ranked a drum-loop candidate at rank 3+3=6,
the guard fails and the bass role can still redirect away from Drums.

---

## 7. Every Branch That Can Turn Pitched Instruments Into Human/Voice FX

### `_measured_broad_bucket_for_smoke_stability()` — `vocal_phrase` shape gate (~line 290)
Already covered in §4. The key additional risk: this method runs **before**
`_true_bucket_for_candidate_conflict()`. The exemption
`measured_role in {pitched_music_loop, pitched_music_phrase, ...}` may not fire if the
measured role is a softer classification like `clean_sustained_tonal_instrument_loop`.

### `_early_candidate_adjudication()` — `weak_voice_false_positive` block (~line 856)
```python
weak_voice_false_positive = bool(
    raw.final_top == "FX"
    and "human and voice" in raw_path
    and instrument_phrase_strength >= 0.84
    and not positive_voice_claim
    and not direct_one_shot_voice_claim
)
if weak_voice_false_positive:
    best_instrument = self._best_candidate(raw, include_top={"Instruments"}, ...)
    if best_instrument is not None:
        return self._redirect_from_raw(raw, instrument_path, "stable pitched instrument claim blocked weak Human/Voice false positive")
    return self._redirect_from_raw(raw, "Instruments/Instrument Loops/Loops", ...)
```
This block **correctly** redirects sax/piano away from Human/Voice FX. The risk is the
**fallthrough**: if `instrument_phrase_strength < 0.84` or `positive_voice_claim` is
barely true (built from the loose `formant_hit_voice_claim` path in §4), the block is
skipped and the instrument lands in Human/Voice FX anyway.

### `_early_candidate_adjudication()` — the reed-like phrase rescue (~line 987)
```python
if role in {pitched_reed_or_instrument_loop, pitched_reed_or_instrument_phrase}:
    if sax_claim and raw_fx_false_positive:
        return self._broaden_from_raw(raw, eligibility, "measured sax/woodwind claim beat non-reed FX...")
    if best_reed is None or best_reed > raw_score + 10.0:
        return self._review_from_raw(...)
```
This correctly sends sax to `Instruments/Brass and Woodwinds`, but the inner
`else` path (when the reed role is present but no reed candidate exists) goes to
**review** rather than staying in Instruments. Instruments/Instrument Loops/Loops would be a
safer landing than review for a clear reed-measured loop.

### `same_family_role_bucket_decision()` — `formant_hit_voice_claim` path
Documented in §4. The threshold `winner_role_fit >= 0.55` on `voiced_one_shot` is the
primary mechanism by which piano/synth one-shots with body-view formant character land in
a Human/Voice broad bucket synthesised from a **synthetic row** (`broad_measured_role_decision`)
with no actual Human/Voice candidate required.

---

## 8. Proposed Refactor Plan

### Principle

**No module may construct a `ConsensusDecision` with a real `folder_path` except `FamilyClaimArbiter`.**

All other modules return evidence: voter rankings, role strengths, shape labels, typed claims.
The arbiter collects all competing claims, picks the winner by a single comparator, then
calls `PlacementResolver` once.

This does **not** require rewriting the whole app. The rescue logic in `_early_candidate_adjudication`
and `_measured_broad_bucket_for_smoke_stability` is largely correct. The problem is the *output
type*, not the logic. Changing the return type from `ConsensusDecision` to `FamilyClaim` and
consolidating the final decision in one place removes most of the regression surface.

---

### Step 1 — Introduce `ConsensusClaim` (new intermediary type)

Create this in `family_claims.py` alongside the existing `FamilyClaim`:

```python
@dataclass(frozen=True)
class ConsensusClaim:
    """Typed evidence object returned by any pre-arbiter module.
    
    No module except FamilyClaimArbiter may convert this to a ConsensusDecision.
    """
    family: str           # "Drums" | "Instruments" | "FX" | "Voice" | "_REVIEW"
    sub_family: str       # "Drum Loops" | "Bass Loops" | "Human and Voice FX" | etc.
    strength: float       # 0.0–1.0; used as claim priority by the arbiter
    can_override: bool    # True = this claim may move away from the raw family
    reason: str           # diagnostic string
    source: str           # "role_sanity" | "shape_sanity" | "drum_rescue" | etc.
    raw_candidate_score: float | None  # combined_rank_score of the underlying candidate, if any
```

`sub_family` replaces the hardcoded folder-path strings. `PlacementResolver` owns the mapping
from `(family, sub_family)` → actual folder string.

---

### Step 2 — Change all pre-arbiter methods to return `ConsensusClaim | None`

**In `consensus.py`**, change the return type of these methods:

| Method | Current return | New return |
|---|---|---|
| `top_family_sanity_decision()` | `ConsensusDecision | None` | `ConsensusClaim | None` |
| `concrete_fx_gate_override()` | `ConsensusDecision | None` | `ConsensusClaim | None` |
| `shape_sanity_decision()` | `ConsensusDecision | None` | `ConsensusClaim | None` |
| `role_sanity_decision()` | `ConsensusDecision | None` | `ConsensusClaim | None` |
| `broad_measured_role_decision()` | `ConsensusDecision | None` | `ConsensusClaim | None` |
| `same_family_role_bucket_decision()` | `ConsensusDecision | None` | `ConsensusClaim | None` |
| `prefer_profile_leaf_with_broad_support()` | `ConsensusDecision | None` | `ConsensusClaim | None` |

Change `ConsensusRunner.choose()` to return a new `RawConsensusBundle`:

```python
@dataclass
class RawConsensusBundle:
    winner: dict[str, Any] | None         # best shared candidate row
    shared: list[dict[str, Any]]          # all shared candidates
    claims: list[ConsensusClaim]          # every claim produced internally
    early_exit_review: ConsensusDecision | None  # only for broken/tiny/no-consensus
```

`choose()` no longer returns `ConsensusDecision` directly. It accumulates all `ConsensusClaim`
objects and returns the bundle. The only early exits that still return `ConsensusDecision`
are the broken/tiny and no-voter-consensus cases (lines 121–128 in the current code),
because those are not family placement decisions.

**In `decision_core_v2.py`**, change the return type of these methods:

| Method | Current return | New return |
|---|---|---|
| `_redirect_from_raw()` | `ConsensusDecision` | `ConsensusClaim` |
| `_broaden_from_raw()` | `ConsensusDecision` | `ConsensusClaim` |
| `_review_from_raw()` | `ConsensusDecision` | `ConsensusClaim` (family="_REVIEW") |
| `_review_from_raw_with_role()` | `ConsensusDecision` | `ConsensusClaim` (family="_REVIEW") |

Change `DecisionCoreV2.apply_eligibility()` to return `list[ConsensusClaim]` instead of
`ConsensusDecision`. Rename it to `gather_eligibility_claims()`.

Change `DecisionCoreV2.choose()` to:
1. Call `raw_consensus.choose()` → `RawConsensusBundle`
2. Call `gather_eligibility_claims()` → `list[ConsensusClaim]`
3. Pass both to `FamilyClaimArbiter.adjudicate()` → `ConsensusDecision`

---

### Step 3 — Create `FamilyClaimArbiter`

New file: `src/aaron_sound_sorter/engine/family_claim_arbiter.py`

```python
class FamilyClaimArbiter:
    """The only place that converts claims into a final ConsensusDecision."""
    
    def __init__(self, placement_resolver: PlacementResolver) -> None:
        self.resolver = placement_resolver
    
    def adjudicate(
        self,
        bundle: RawConsensusBundle,
        eligibility_claims: list[ConsensusClaim],
        facts: SharedAudioFacts,
    ) -> ConsensusDecision:
        if bundle.early_exit_review is not None:
            return bundle.early_exit_review  # only legitimate early exit
        
        all_claims = bundle.claims + eligibility_claims
        raw_claim = self._claim_from_winner(bundle.winner, bundle.shared)
        
        # Arbiter rule: a claim may override the raw winner only when
        # can_override=True AND it was produced from actual candidate evidence
        # (raw_candidate_score is not None) OR it is a review signal.
        winner_claim = self._pick_winner(raw_claim, all_claims)
        folder_path = self.resolver.resolve(winner_claim)
        return ConsensusDecision(
            final_label=winner_claim.sub_family,
            final_top=winner_claim.family,
            folder_path=folder_path,
            consensus_status=winner_claim.source,
            reason=winner_claim.reason,
            ...
        )
    
    def _pick_winner(
        self,
        raw: ConsensusClaim,
        claims: list[ConsensusClaim],
    ) -> ConsensusClaim:
        # Single comparator: strongest can_override=True claim with
        # raw_candidate_score <= raw.raw_candidate_score + OVERRIDE_MARGIN
        # wins; otherwise raw wins.
        # All tie-breaking happens here, in one place.
        ...
```

The comparator in `_pick_winner` replaces the 14 sequential `if decision is not None: return`
gates in `ConsensusRunner.choose()` and the 63 call-sites in `DecisionCoreV2`.

---

### Step 4 — Replace hardcoded folder strings with `PlacementResolver`

Move all 26 hardcoded path strings out of `decision_core_v2.py` and into
`PlacementResolver.resolve(claim: ConsensusClaim) -> str`.

The resolver holds a single lookup table:

```python
FAMILY_FOLDER_MAP = {
    ("Drums", "Drum Loops"):           "Drums/Drum Loops/Loops",
    ("Drums", "Kick One Shot"):        "Drums/Kick Drums/Generic Kick/One Shots",
    ("Drums", "Percussion One Shot"):  "Drums/Percussion/Generic Percussion/One Shots",
    ("Instruments", "Bass Loops"):     "Instruments/Bass/Bass Loops",
    ("Instruments", "Instrument Loops"): "Instruments/Instrument Loops/Loops",
    ("Instruments", "Brass Woodwinds"): "Instruments/Brass and Woodwinds/Loops",
    ("FX", "Human and Voice FX"):      "FX/Human and Voice FX",
    ("_REVIEW", ""):                   "_TO_REVIEW/Measured Role Conflict",
    ...
}
```

Future folder renames touch this table once, not 26 locations.

---

### Step 5 — Fix the three specific threshold problems

These are not threshold tuning — they are type-safety fixes in the claim production code:

**Fix A: `formant_hit_voice_claim` gate in `same_family_role_bucket_decision()`**  
Raise `winner_role_fit >= 0.55` to `winner_role_fit >= 0.75`.  
At 0.55, piano and synth one-shots with formant-like spectral profiles pass. At 0.75, only
samples that actually dominant-score as `voiced_one_shot` pass.  
**Also:** this path currently calls `broad_measured_role_decision()` which produces a
**synthetic** bucket with no real candidate. After the refactor, it should instead only emit
a `ConsensusClaim(can_override=False)` when no actual Human/Voice candidate exists in
`shared`. `can_override=False` means the arbiter will not use it to override the raw winner.

**Fix B: `bass_rescue_role` over-broad OR condition in `_early_candidate_adjudication()`**  
```python
# Current (too broad):
bass_rescue_role = (
    eligibility.role_name == "bass_loop"
    or measured_role == "bass_loop"    # fires regardless of eligibility confidence
    or (shape == "bass_phrase" ...)
)

# Proposed:
bass_rescue_role = (
    eligibility.role_name == "bass_loop"
    or (measured_role == "bass_loop" and eligibility.confidence >= 0.78)  # require confirmation
    or (shape == "bass_phrase" ...)
)
```
This makes the `measured_role == "bass_loop"` path require the same eligibility confidence
gate that the other arms already require.

**Fix C: Human/Voice rescue with no actual candidate — convert to `can_override=False` claim**  
`_broad_voice_bucket_for_non_voice_instrument()` and the `same_family_role_bucket_decision()`
voice path currently produce real folder destinations from role evidence alone. After the
refactor, when no `has_voice_candidate=True` Human/Voice claim exists (as checked in
`build_human_voice_claim()`), these paths should produce `ConsensusClaim(can_override=False)`.
The arbiter then cannot use them to move the file across families.

---

## 9. What to Leave Unchanged

The following are architecturally sound and should not be altered during this refactor:

- **`family_claims.py`** — `FamilyClaim` and `build_human_voice_claim()` are the correct pattern. `build_human_voice_claim()` is already called in `_early_candidate_adjudication()`. Extend this pattern to `build_drum_loop_claim()`, `build_bass_loop_claim()`, `build_instrument_claim()` as the claim producers.
- **`ConsensusRunner.shared_candidates()`** — correct; returns evidence, not decisions.
- **`ConsensusRunner.prefer_dynamic_broad_role_candidate()`**, **`prefer_profile_role_candidate()`**, **`prefer_cross_family_role_candidate()`** — these select the best shared candidate row within the same family, not across families. They are candidate-selection helpers, not placement deciders. Keep them as-is; they feed into the `RawConsensusBundle.winner`.
- **`DecisionCoreV2._best_candidate()`**, **`_best_candidate_score()`**, **`_has_close_candidate()`** — evidence-reading utilities; correct as-is.
- **The full/direct body role evidence reading helpers** (`_role_strength_from_facts`, `_direct_body_role_strength_from_facts`, etc.) — correct as-is.

---

## 10. Summary of Changes by File

| File | Change |
|---|---|
| `family_claims.py` | Add `ConsensusClaim` dataclass; add `build_drum_loop_claim()`, `build_bass_loop_claim()`, `build_instrument_claim()` helpers |
| `consensus.py` | Change 7 method return types from `ConsensusDecision | None` to `ConsensusClaim | None`; change `choose()` to return `RawConsensusBundle` |
| `decision_core_v2.py` | Change 4 terminal methods to return `ConsensusClaim`; change `apply_eligibility()` to return `list[ConsensusClaim]`; apply Fix A, B, C from §8 |
| `family_claim_arbiter.py` (new) | `FamilyClaimArbiter.adjudicate()` — single `ConsensusDecision` constructor for the entire system |
| `placement_resolver.py` (new or extend) | Move all 26 hardcoded folder strings into `FAMILY_FOLDER_MAP` |
| No changes needed | `roles.py`, `role_gate.py`, `physics_voter.py`, `brain_voter.py`, `eligibility.py` |

The refactor is a type-level change to the output of existing methods plus one new class.
The rescue logic, threshold values (other than Fix A/B/C), and candidate-evidence reading are
preserved exactly. Test coverage should be satisfiable by checking that `ConsensusDecision`
is only constructed inside `FamilyClaimArbiter.adjudicate()` — that can be enforced with a
single grep-based CI check.
