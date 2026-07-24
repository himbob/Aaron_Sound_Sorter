# Research Basis for Adaptive Category Tolerance

The implementation follows a conservative engineering conclusion from musical-audio
research: source identity and absolute pitch are related but not identical, and timbre
also varies with velocity, articulation, register, playing style, room, processing,
and instrument subtype.

Relevant primary research themes:

- NSynth models instrument notes across instrument identity, pitch, velocity, and
  acoustic qualities, and its learned space supports timbral interpolation.
- Pitch/timbre disentanglement research explicitly learns separate latent spaces for
  instrument identity and pitch.
- Pitch-spiral instrument-recognition work states that classification must tolerate
  transformations in pitch, nuance, and expressive technique.
- Instrument-recognition work combines pitch and timbre rather than treating either
  one as a complete identity signal.
- Percussion research and standard audio descriptors rely on temporal attack/decay,
  spectral distribution, noise/tonality, and resonant behavior—not one Hertz cutoff.

Engineering consequence for this sorter:

1. Do not use absolute register as the dominant category identity feature.
2. Do not erase pitch and harmonic evidence entirely; they still help distinguish
   contradictory families.
3. Learn which measured dimensions vary inside each human-taught target.
4. Preserve dimensions that remain stable as automatic guardrails.
5. Keep local prototypes for multimodal categories rather than averaging all examples
   into one centroid.
6. Require multiple examples before claiming category-wide learned variability.
