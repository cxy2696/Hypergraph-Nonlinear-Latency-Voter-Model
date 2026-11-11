# Data: provenance, preprocessing and limitations

## 1. Source and licence

| field | value |
|---|---|
| dataset | *Social Media Sentiments Analysis Dataset* |
| author (Kaggle) | Kashish Parmar |
| URL | <https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset> |
| licence | **CC0: Public Domain** (declared on the Kaggle page) |
| file | [`sentimentdataset.csv`](../sentimentdataset.csv) |
| bytes | 170 776 |
| SHA-256 | `725c6f7bc928c2bf741094f37078024b305982523fe6d0b8cb20a6cf5275c63b` |
| rows × columns | 732 × 15 |
| time span | 2010-05-15 15:30 to 2023-10-22 20:45 |

The file in this repository is byte-identical to the Kaggle archive
(170 776 bytes), so the corpus is included under CC0. The audit in
[`ERRATA.md`](ERRATA.md) §A1–A4 concerns how v1 *used* the file, not the file
itself.

## 2. Schema

The CSV has 15 columns; all are non-null.

| column | type | role in v2 |
|---|---|---|
| `Unnamed: 0`, `Unnamed: 0.1` | integer | ignored (row indices) |
| `Text` | string | ignored |
| `Sentiment` | string | emotion label → binary valence |
| `Timestamp` | string | metadata only |
| `User` | string | node identifier (stripped) |
| `Platform` | string | metadata only (`Instagram` 258, `Twitter` 243, `Facebook` 231) |
| `Hashtags` | space-separated string | hyperedge construction |
| `Retweets`, `Likes` | numeric | node weight (engagement) |
| `Country` | string | metadata only (33 values) |
| `Year` … `Hour` | integer | metadata only |

## 3. Preprocessing

The pipeline is implemented in [`hnl/data.py`](../hnl/data.py).

1. **Strip and collapse whitespace** in `User`, `Sentiment`, `Platform` and
   every hashtag token.
2. **Merge user identities.** `"User123 "` and `"User123"` become one node.
   This reduces 685 raw identifiers to **670 users**; 617 users have one post,
   44 have two, 9 have three.
3. **Normalise hashtags.** Case-fold and drop the leading `#`; a token is
   counted at most once per post.
4. **Build hyperedges.** For each hashtag, the set of users who used it. Keep
   groups of size $\ge2$, drop duplicate node ids, merge identical groups.
5. **Scope.**
   * `full`: all 670 users; isolated users are kept but frozen.
   * `active`: only users in at least one hyperedge (491 users).
   * `lcc`: the largest connected component of the active graph (140 users).
6. **Weights.** `engagement = Retweets + Likes` averaged per user, divided by
   the maximum (so weights lie in $(0,1]$), floored at $10^{-3}$. Only relative
   weights matter, because $\pi_e$ is invariant to a common rescaling.

## 4. Sentiment → binary opinion

The `Sentiment` column contains **191 distinct labels** after whitespace
normalisation (the raw column has 279 whitespace variants). It is an emotion
taxonomy, not a positive/negative stance label. Three mappings are available:

| mapping | rule | users with an opinion |
|---|---|---|
| `valence` (default) | documented lexicon: 115 positive, 54 negative, 22 ambiguous; ambiguous labels are unmapped | 576 / 670 (full), 422 / 491 (active) |
| `paper` | `Sentiment == "Positive"` → 1, everything else → 0 | 670 / 670 |
| `strict` | only the literal labels `Positive` / `Negative`; everything else unmapped | ~49 / 670 |

Post-level counts under `valence`: 450 positive, 179 negative, 103 unmapped.
At user level the mapped positive share is **0.670** in the active graph.

The lexicon lives in [`hnl/lexicon.py`](../hnl/lexicon.py) and is fixed; it was
specified before any simulation output was inspected. It is a modelling choice,
not a gold standard. `paper` and `strict` are provided so that the sensitivity
to this choice can be checked. Users whose labels are all ambiguous are filled
from the empirical positive share when `init="data"` is requested; the number
of such users is reported in every run manifest.

## 5. Graph structure

| scope | nodes | hyperedges | coverage | isolated | components | largest component | largest share | mean \|e\| | mean degree (covered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 670 | 171 | 73.3 % | 179 | 260 | 140 | 20.9 % | 3.46 | 1.20 |
| active | 491 | 171 | 100 % | 0 | 81 | 140 | 28.5 % | 3.46 | 1.20 |
| lcc | 140 | 45 | 100 % | 0 | 1 | 140 | 100 % | 4.16 | 1.34 |

Hyperedge sizes range from 2 to 15 (mean 3.46). Most hashtags do not create a
group at all: of 975 distinct hashtags, **802 are used by exactly one user**;
173 are multi-user groups, which collapse to 171 unique hyperedges after
merging identical user sets.

The 2-section graph is highly fragmented: 260 components in the full graph and
81 in the active graph. In the full graph, 179 of 670 nodes have no hyperedge
at all; the active scope removes them, but the largest component still contains
only 28.5 % of the active population. Section 6 of [`MODEL.md`](MODEL.md) shows why this matters: the
process absorbs into a **component-wise mosaic**, not a global consensus.

## 6. What the data can and cannot support

**Can support**

* a realistic, sparse group structure for a structural simulation;
* a check that the model's long-run behaviour is dominated by fragmentation;
* a demonstration of how sensitive conclusions are to preprocessing choices
  (deduplication, whitespace, valence mapping, scope).

**Cannot support**

* **Dynamical validation.** There is no per-user opinion trajectory: 617 of 670
  users appear once, and the model's time index is a sweep, not the timestamp.
  The paper's "comparison with real-world data" is therefore a comparison with
  a static graph, not with observed dynamics.
* **Platform comparisons.** Platform is pooled; the model has no platform
  parameter.
* **Representativeness.** 732 posts with pseudonymous handles are a convenience
  sample, not a platform-scale or representative sample.
* **Stance measurement.** Emotion categories are not positions on a binary
  issue; the valence projection is coarse and partly ambiguous
  (22 of 191 labels are explicitly unmapped).
* **Influence measurement.** Static retweets and likes are engagement snapshots,
  not estimates of social influence.

These limitations are reflected in the wording of
[`RESULTS.md`](RESULTS.md) and in the run manifest, which records the file hash,
row counts, mapping, scope and mapping coverage.

## 7. Regenerating the structural report

```bash
python -m hnl.cli diagnose --csv sentimentdataset.csv --scope full \
    --out results/diagnostics.md
```

The command writes the counts above plus the top hashtags and a structural
implication paragraph. The full machine-readable metadata is available via
`--json results/diagnostics.json`.

## 8. Citation

```bibtex
@misc{parmar_social_media_sentiments,
  author       = {Parmar, Kashish},
  title        = {Social Media Sentiments Analysis Dataset},
  howpublished = {Kaggle},
  year         = {2024},
  note         = {CC0: Public Domain},
  url          = {https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset}
}
```
