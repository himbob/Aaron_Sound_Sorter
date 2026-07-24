# Archived: Smoke Test Analysis - June 6, 2026

**Analysis Date:** June 6, 2026  
**Test Scope:** FX and Instrument Smoke Tests  
**Data Availability:** ✓ FX Test Complete | ⚠️ Instrument Test Results Not Found

---

## Part 1: FX Smoke Test Analysis

### Test Directory
- **Source:** `_real_sort_tests/fx_20260606_133835/` (Created: 13:38:34)
- **Results Output:** `_pytest_outputs/v26_fx_smoke_reverb/`
- **Sample Count:** 3 samples
- **Test Timestamp:** June 6, 2026 @ 13:36 UTC

### 1. Structure Overview & Case Enumeration

| # | Sample Name | Duration | Final Label | Final Category | Status | Mutators |
|---|---|---|---|---|---|---|
| 1 | ABOUTME_94_DRUMLOOP.wav | 2.40s | Drums/Drum Loops/Loops | **Drums** | ✅ Correct | 0 |
| 2 | HipHopTapes_29_Saxophone_D#m_90bpm.wav | 2.67s | Instruments/Woodwinds/Saxophone/Loops | **Instruments** | ✅ Correct | 1 |
| 3 | SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav | 10.66s | Instruments/Woodwinds/Saxophone/Loops | **Instruments** | ✅ Correct | 1 |

**Success Rate: 3/3 (100%)**  
**No _TO_REVIEW failures detected**

---

### 2. Detailed Voter Log Analysis

#### Case 1: ABOUTME_94_DRUMLOOP.wav → Drums/Drum Loops/Loops ✅

**Consensus Status:** `strong_consensus`  
**Decision Reason:** "brain and physics shared the winning category"

| Voter | Initial Vote | Confidence/Rank | Note |
|---|---|---|---|
| **Brain Ensemble** | FX/Textures/Natural Ambience/Fire/One Shots | Rank 5 | Wrong initial direction |
| **Physics Voter** | Drums/Drum Loops/Loops | Rank 1 (physics winner) | **Correct detection** |
| **Shape Voter** | beat_loop | Confidence: 0.8195 | Strong drum pattern recognition |

**Raw Consensus Path:**  
1. Brain voted FX (textures) - incorrect
2. Physics voted Drums - correct
3. **Arbiter chose:** Physics winner (Drums) ✓
4. Final placement: Drums/Drum Loops/Loops

**Physics Evidence Strength:**
- Drum loop structure clearly identified
- Shape confidence: 0.8195 (strong beat pattern)
- No voice interference
- No competing pitched evidence

---

#### Case 2: HipHopTapes_29_Saxophone_D#m_90bpm.wav → Instruments/Woodwinds/Saxophone/Loops ✅

**Consensus Status:** `final_measured_sax_loop_invariant`  
**Decision Reason:** "final structure invariant kept measured sax/reed loop body out of FX/review/generic sibling outcomes"

| Voter | Initial Vote | Confidence/Rank | Note |
|---|---|---|---|
| **Brain Ensemble** | Instruments/Brass/Trumpet/One Shots | Rank 3 | Wrong instrument family |
| **Physics Voter** | Instruments/Woodwinds/Saxophone/One Shots | Rank 6 | **Correct instrument** |
| **Shape Voter** | solo_phrase | Confidence: 0.9662 | Strong phrase pattern |

**Raw Consensus Path:**
1. Brain voted Instruments/Brass/Trumpet (wrong subfamily)
2. Physics voted Instruments/Saxophone (correct)
3. Raw winner: Instruments/Acoustic Guitar/One Shots (consensus)
4. **Post-winner mutation:** `protect_measured_voice_from_non_voice_leaf` applied
5. Final placement: Instruments/Saxophone/Loops ✓

**Key Observation:**
- Mutation count: 1 (protective mutation)
- The sax loop invariant guard specifically protected the measured saxophone body
- Brain confused sax with trumpet, but physics correctly identified woodwind family
- Shape confidence was very high (0.9662) confirming phrase structure

---

#### Case 3: SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav → Instruments/Woodwinds/Saxophone/Loops ✅

**Consensus Status:** `final_measured_sax_loop_invariant`  
**Decision Reason:** "final structure invariant kept measured sax/reed loop body out of FX/review/generic sibling outcomes"

| Voter | Initial Vote | Confidence/Rank | Note |
|---|---|---|---|
| **Brain Ensemble** | FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX | Rank 1 | **Wrong family** |
| **Physics Voter** | Instruments/Voice/Choir/One Shots | Rank 7 | Wrong instrument type |
| **Shape Voter** | pitched_repetition_phrase | Confidence: 0.9041 | Weak for pitched instrument detection |

**Raw Consensus Path:**
1. Brain voted FX (Synth Riser) - INCORRECT cross-family error
2. Physics voted Instruments/Voice (wrong subfamily)
3. Raw winner: Instruments/Instrument Loops/Loops (consensus fallback)
4. **Post-winner mutation:** `protect_measured_voice_from_non_voice_leaf` applied
5. Final placement: Instruments/Saxophone/Loops ✓

**CRITICAL OBSERVATION:**
- **Routing Error:** Brain initially routed saxophone into FX (Synth Riser) ❌
- **Physics Weakness:** Misidentified as voice/choir instead of saxophone
- **Recovery:** Only successful because sax loop invariant caught this cross-family error
- **Risk Indicator:** Without the protective invariant, this would have gone to FX
- Mutation count: 1 (necessary protective correction)

---

## 3. Pattern Detection (FX Test)

### ✅ Pattern 1: Strong Consensus Drums (Case 1)
- **Trigger:** Clear drum loop structure
- **Physics Performance:** Excellent (rank 1)
- **Brain Performance:** Failed (voted FX textures)
- **Recovery:** Strong consensus arbiter ruled physics winner
- **Status:** ✅ Working correctly

### ⚠️ Pattern 2: Brain Cross-Family Routing Error (Case 3)
- **Issue:** Brain routed saxophone into FX/Synth Riser
- **Why:** Long repetitive structure + energy profile similar to synth risers
- **Detection:** Shape confidence was 0.9041, but physics chose wrong subfamily (voice)
- **Recovery:** Sax loop invariant guard caught it
- **Risk:** Without this guard, would achieve only 67% success (2/3)

### ⚠️ Pattern 3: Physics Subfamily Confusion (Case 2 & 3)
- **Case 2:** Physics said Trumpet, correct family but wrong subfamily
- **Case 3:** Physics said Voice/Choir instead of Saxophone
- **Root Cause:** Reed/woodwind vs. voice acoustic signatures overlap
- **Impact:** Not severe (family correct) but shows weak discrimination in physics voter

---

## 4. Categorization Success Rate (FX Test)

### By Major Category
| Category | Cases | Correct | % | Issues |
|---|---|---|---|---|
| **Drums** | 1 | 1 | 100% | None |
| **Instruments** | 2 | 2 | 100% | Brain routing attempt; Physics subfamily confusion |
| **FX** | 0 | 0 | N/A | N/A |

### By Subcategory
| Subcategory | Cases | Correct | % | Notes |
|---|---|---|---|---|
| Drums/Drum Loops | 1 | 1 | 100% | Consensus decision, physics rank 1 |
| Instruments/Saxophone | 2 | 2 | 100% | ⚠️ Both needed post-winner mutation |
| Instruments/Woodwinds | 2 | 2 | 100% | One was misidentified as voice initially |

**Caveats:**
- FX cases: 0 test cases (cannot evaluate)
- Instrument precision: 100% for final placement, but 50% for physics initial ranking

---

## 5. Voter Evidence Quality Analysis

### Physics Voter Performance

#### Strengths:
1. **Drum detection:** Rank 1 for drum loop (Case 1)
2. **Family identification:** All 3 correctly placed in families
3. **Shape agreement:** Aligned with shape voter on drum patterns

#### Weaknesses:
1. **Reed/Wind discrimination:** 
   - Case 2: Said "Trumpet" instead of "Saxophone" (wrong subfamily)
   - Case 3: Said "Voice/Choir" instead of "Saxophone" (very wrong)
2. **Cross-family risk:** Misidentified saxophone as voice
3. **Ranking inconsistency:** 
   - Drum case: Rank 1
   - Sax cases: Rank 6-7 (low confidence)

### Shape Voter Performance

#### Observations:
1. **Beat/Loop recognition:** 0.8195 confidence for drum loop (good)
2. **Phrase detection:** 0.9662 confidence for solo phrase (excellent)
3. **Repetitive pattern:** 0.9041 confidence for pitched repetition (good)
4. **Consistency:** All confidence scores in strong range (0.80+)

### Brain Ensemble Performance

#### Critical Failures:
1. **Case 1:** Voted FX Textures for drum loop (completely wrong)
2. **Case 3:** Voted FX Synth Riser for saxophone (cross-family error)
3. **Case 2:** Voted Trumpet instead of Saxophone (subfamily only)

#### Analysis:
- **Success Rate:** 1/3 correct initial placement (33%)
- **Family Error Rate:** 1/3 (cross-family to FX)
- **Subfamily Error Rate:** 3/3 wrong initially (all needed correction)

**Concern:** Brain ensemble being saved entirely by post-winner invariants, not by correct voting

---

## Part 2: Instrument Smoke Test Analysis

### Status: ⚠️ DATA NOT FOUND

**Source Directory:** `_real_sort_tests/inst_20260606_134242/` (Created: 13:42:43)  
**Expected Results:** Would be in `_pytest_outputs/` or `_reports/`

**Search Results:**
- ❌ No manifest.csv found for this test
- ❌ No debug_packets.jsonl found for this test  
- ❌ No corresponding entry in _pytest_outputs/ or _reports/
- ⚠️ `_reports/inst_smoke_suspect_panel/` exists but dated June 5 (not June 6)

**Possible Reasons:**
1. Test was staged but not actually executed
2. Results are in a different location (custom output path)
3. Test is pending/in-progress
4. Results were manually moved or archived

**Recommendation:** Clarify whether the inst_20260606_134242 test was run and where to find its results.

---

## 6. Recommended Fixes (Based on FX Test Findings)

### Priority 1: Physics Voter Woodwind/Voice Discrimination

**Problem:** Physics voter confuses saxophone with voice/choir  
**Location:** `voters/physics_voter.py` - Reed/voice scoring layer  
**Evidence:** Case 3 scored saxophone as "Instruments/Voice/Choir"

**Fix Strategy:**
```
In physics_layer evaluation for woodwind detection:
- Add explicit pitch stability check
- Saxophone: sustained pitch, clear partials, low vibrato
- Voice: pitch drift, formant structure, natural vibrato
- Threshold: If pitch_stability >= 0.80 AND vibrato_natural < 0.40, prefer woodwind
```

**Expected Impact:** Reduce subfamily confusion, improve initial physics ranking for saxophones

---

### Priority 2: Brain Ensemble Cross-Family Routing to FX

**Problem:** Brain routed 2-bar saxophone phrase into FX/Synth Riser  
**Location:** `voters/brain_voter.py` - Domain adaptation layer  
**Root Cause:** Repeated tonal patterns + long duration scored as "synthesis riser"

**Fix Strategy:**
```
In brain ensemble FX/Synth Riser detection:
- Add gate: If pitch_stability >= 0.75 AND note_changes > 3, return None (not synth riser)
- Synth risers: smooth upward motion, minimal note changes
- Sax phrases: multiple pitch changes, note articulations visible
- Exclude measured-role="reed_wind_loop" from synth riser consideration
```

**Expected Impact:** Prevent saxophone→FX routing, reduce need for protective invariants

---

### Priority 3: Strengthen Physical Evidence for Saxophone Detection

**Problem:** Physics voter weak on saxophone identification (rank 6-7)  
**Location:** `voters/physics_layer_types.py` - Brass/Woodwind layer

**Current Evidence Quality:**
- Case 2: Identified correct family but wrong subfamily (Trumpet vs Saxophone)
- Case 3: Failed completely (Voice instead of Saxophone)

**Fix Strategy:**
```
Enhance reed_wind_score calculation:
1. Add spectral centroid analysis (woodwind characteristic dip at ~3-5kHz)
2. Add formant tracking (reed winds have stable formants, voices have mobile formants)
3. Add tremolo/trill detection (characteristic of woodwind technique)
4. Threshold improvements:
   - reed_wind_score >= 0.80 should rank Rank 1-2 (not 6-7)
   - Confidence multiplier when shape="solo_phrase" AND pitch_stability >= 0.75
```

**Expected Impact:** Improve physics ranking for woodwinds from "rank 6-7" to "rank 1-3"

---

### Priority 4: Reduce Dependency on Post-Winner Protective Invariants

**Problem:** FX test passing only because of `final_measured_sax_loop_invariant`  
**Current Status:** Without this guard, only 67% success (2/3)

**Analysis:**
- Case 1: No mutation needed (physics correct initially)
- Case 2: Mutation needed (brain wrong, protected)
- Case 3: Mutation CRITICAL (brain routed to FX, protected)

**Fix Strategy:**
```
Instead of relying on post-winner guards:
1. Fix physics voter (Priority 1-3 above)
2. Once physics improves, re-evaluate if invariant still needed
3. Monitor: How many test cases would fail without this specific invariant?
```

**Risk:** Current success relies on invariant, not on correct voting

---

## 7. Threshold Harmonization Status

Based on session memory (`/memories/session/plan.md`), previous fixes were applied:

| Threshold | Location | Current Value | Status |
|---|---|---|---|
| Voice structural check | measured_early_preflight | 0.75 | ✅ Unified |
| Low rhythmic drum redirect | measured_early_drum_fx | 0.75 | ✅ Applied |
| Sustain guard (synth) | measured_early_drum_fx | ≥ 0.50 | ✅ Applied |
| Pitch confidence guard | measured_early_drum_fx | ≥ 0.72 | ✅ Applied |

**FX Test Result:** These fixes appear to be working (100% success)

**Question for Instruments Test:** Need data to verify if these thresholds also improve instrument routing (especially saxophone vs. voice conflicts)

---

## Summary & Next Steps

### FX Test Conclusions
- ✅ **Success Rate:** 100% (3/3 correct)
- ✅ **No _TO_REVIEW failures**
- ⚠️ **Quality Concerns:** 
  - 2/3 cases needed post-winner mutations
  - Brain voter accuracy only 33%
  - Physics voter subfamily confusion on non-drums

### Top 3 Failing Patterns Identified (from voter behavior, not failures)
1. **Physics Woodwind/Voice Confusion** (Priority 1)
2. **Brain Cross-Family to FX Routing** (Priority 2)  
3. **Physics Weak Ranking for Non-Drums** (Priority 3)

### Missing Data
- ❌ **Instrument Smoke Test results needed** - Cannot analyze without data
- Required: manifest.csv + debug_packets.jsonl for inst_20260606_134242

### Recommended Immediate Actions
1. **Locate instrument test results** or confirm if test was executed
2. **Implement Priority 1 fix** (Physics woodwind/voice discrimination)
3. **Monitor:** Test again to verify Physics improvement
4. **Then assess:** Priority 2 fix necessity based on updated results
