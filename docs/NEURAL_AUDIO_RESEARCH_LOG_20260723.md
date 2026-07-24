# Neural Audio Research — Search Execution Log

Research date: 2026-07-23

## Exact search queries executed

1. `LAION AI CLAP official GitHub repository audio embeddings`
2. `Hugging Face CLAP model documentation audio embeddings`
3. `Microsoft BEATs official repository audio pretraining`
4. `MERT music understanding official repository embeddings`
5. `Essentia TensorFlow musicnn embeddings official models documentation`
6. `OpenL3 audio embeddings official documentation tutorial`
7. `scikit-learn probability calibration isotonic logistic regression official user guide`
8. `HDBSCAN official documentation clustering embeddings`
9. `site:essentia.upf.edu models musicnn embeddings TensorFlow official`
10. `site:hdbscan.readthedocs.io parameter selection official HDBSCAN`
11. `site:faiss.ai getting started cosine similarity vector search official`
12. `site:onnxruntime.ai execution providers CPU CoreML official`
13. `Essentia official models musicnn embedding official`
14. `FAISS official getting started similarity search`
15. `librosa feature extraction API official documentation`
16. `Spotify Voyager nearest neighbor official documentation`
17. `Chromaprint official documentation audio fingerprinting AcoustID`
18. `MERT official Hugging Face model card m-a-p MERT v1 95M`
19. `site:huggingface.co laion larger_clap_music_and_speech model card`
20. `site:github.com yizhilll MERT official repository license`

## Distinct domains actually fetched and read

1. GitHub — https://github.com/LAION-AI/CLAP
2. Hugging Face — https://huggingface.co/laion/larger_clap_music_and_speech
3. Microsoft Research — https://www.microsoft.com/en-us/research/publication/beats-audio-pre-training-with-acoustic-tokenizers/
4. arXiv — https://arxiv.org/abs/2306.00107
5. OpenL3 docs — https://openl3.readthedocs.io/en/stable/tutorial.html
6. scikit-learn docs — https://scikit-learn.org/stable/modules/calibration.html
7. HDBSCAN docs — https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
8. ONNX Runtime docs — https://onnxruntime.ai/docs/execution-providers/
9. Apple Core ML Tools — https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html
10. Essentia — https://essentia.upf.edu/models.html
11. librosa — https://librosa.org/doc/latest/feature.html
12. Spotify Voyager — https://spotify.github.io/voyager/
13. AcoustID/Chromaprint — https://acoustid.org/chromaprint

## Conclusions used by this implementation

- CLAP supplies audio/text embeddings and music/speech checkpoints.
- The reviewed CLAP music/speech model exposes a 512-dimensional projection and uses
  48 kHz, ten-second input preparation.
- MERT is music-specific but its common model card is non-commercial and requires custom
  model code, so it is not the default production provider.
- Calibration must remain leakage-safe; isotonic calibration is too flexible for small
  datasets, so the initial feedback model is logistic regression.
- HDBSCAN is useful for discovery and noise/outlier analysis, not required for the first
  supervised prototype builder.
- Exact NumPy search is preferable before FAISS or Voyager because the prototype count is
  small and exact behavior is easier to audit.
- ONNX Runtime and Core ML remain deployment options after a model/provider wins on
  held-out accuracy.
- DSP libraries remain valid for objective temporal, tonal, and stereo facts, not semantic
  folder identity.

## Local execution verification — 2026-07-24 UTC

No new web research was required for this verification step. The already
reviewed CLAP checkpoint was executed from a pinned local snapshot:

```text
model: laion/larger_clap_music_and_speech
revision: 195c3a3e68faebb3e2088b9a79e79b43ddbda76b
weight SHA-256: d4e5cf6317c7521ca62c11b524f5646565310e040129fe35719cad661696d745
dimensions: 512
device: CPU
```

The measured trial and its limitations are documented in
`NEURAL_AUDIO_VERIFICATION_HANDOFF_20260724.md`. External model-card claims are
not substituted for the local held-out results.
