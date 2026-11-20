# DarkStack — SimulacraBench submission (team `LLM+`)

Code for the NeurIPS 2026 SimulacraBench competition entry described in
*Stacking in the Dark: Self-Tuning Ensembles for Simulating Survey Respondents
on Unseen Microdata* (Lan Jiang, Dequn Teng; University of Cambridge).

## What this is

A single file, `main.py`, that exposes the competition's required entry point:

```python
predict(frame, schema)  # -> one probability vector per blank PREDICT cell
```

Everything is fitted **inside `predict()`** on the visible respondents the
harness passes in. No LLM, no pretrained weights, no external data, no network
access, no bundled artifact.

Per item, several candidate distributions are built and pooled:

| Candidate | Description |
|-----------|-------------|
| M | crowd marginal |
| L | greedy Dirichlet backoff ladder (closed-form leave-one-out) |
| F | multi-output random forest |
| G | per-item gradient-boosted trees |
| H | wide & deep multi-task student |
| R | H added on top of L's log-probabilities |
| W | multinomial logit |

The candidates are combined by a per-item linear pool fitted by EM on the scored
likelihood, with **bag-size-extrapolated stacking** (weights chosen on a
three-repeat out-of-fold estimate and corrected towards the one-repeat estimate),
followed by structural-zero caps, per-item power sharpening and de-flooring. A
deadline manager scales work to the time budget of each instrument.

## Usage

Package `main.py` (plus `requirements.txt`) at the root of a zip and submit it
to the competition as described in the official starter kit. The code adapts
itself to the development and final phases from the data it is given.

```
pip install -r requirements.txt   # only needed outside the hosted image
```

## Notes

- The file in this repository differs from the version submitted to the Test
  phase only in the module docstring (the internal codename was replaced by
  "DarkStack"); the code is identical.
- `main.py` can optionally blend with the starter kit's `official_baseline.py`
  if that file sits next to it; the blend weight is 0 by default, so the
  submission does not depend on it.
- No competition data is included in this repository.

## License

MIT (see `LICENSE`).
