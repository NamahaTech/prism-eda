# Clustering

`clustering()` asks two questions in order: **is there any group structure
here at all**, and only then, **what are the groups?**

That order is the whole design. Clustering is unfalsifiable by construction —
ask k-means for four groups and it returns four groups, from segmented data and
from uniform noise alike, with a silhouette score, cluster sizes, and a
convincing profile either way. Nothing in the output distinguishes structure
from its absence, so the checks have to come first.

It is a *readiness and segmentation* diagnostic. It returns evidence, findings,
and candidate segment profiles — never a fitted model, and never cluster
assignments to feed downstream.

```python
import prism_eda as pe
from examples.sample_data import customer_segments

result = pe.load({"members": customer_segments()}).clustering()
```

As a one-liner, optionally naming the features:

```python
result = pe.clustering("data/members.csv", features=["spend", "visits", "tenure"])
```

## The verdict

```python
print(result.summary)
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
    print(f"    {finding.summary}")
```

```text
members: 4 candidate segments, but review first. Top issue — Identifier column excluded: member_id. 3 prioritized issue(s) (3 medium). 5 alert(s) describe the geometry.
[medium] Identifier column excluded: member_id
    Near-unique per row; it would add a dimension in which every point is equidistant from every other.
[medium] Constant column excluded: account_status
    Every row holds the same value, so it adds no distance.
[medium] Features differ enormously in scale
    annual_spend spans 1,653x the range of satisfaction. Without standardizing, the distance between two rows is essentially the difference in one column.
[medium] 12 duplicate rows
    2.0% of rows are exact repeats, which weights their location in space more than once.
[medium] The same quantity is present twice
    visits_per_month and visits_per_year correlate at 1.0000, so that quantity carries double weight in the distance.
[low] The data is more clustered than chance
    Hopkins averaged 0.89 (range 0.88-0.91) over 10 samples.
[low] 4 groups reproduce on resampled rows
    k=4 scores 0.55 on silhouette and agrees with itself at 1.00 across 5 resamples.
[low] The groups do not line up with region
    No segment is meaningfully over-represented in any value of region, so that column describes something orthogonal to the grouping.
```

## "No structure" is a real answer

Run the same recipe on uniform noise:

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.uniform(0, 1, (600, 4)), columns=["f0", "f1", "f2", "f3"])

result = pe.load({"noise": noise}).clustering()
print(result.status)
print(result.summary)
```

```text
no_meaningful_structure
noise: no stable cluster structure was found. Partitioning it anyway would divide something continuous.
```

Two findings explain why — Hopkins sits at chance, and no cluster count produced
a partition that reproduces. And crucially:

```python
print("segments profiled:", any(e.kind == "clustering_segments" for e in result.evidence))
print("embedding drawn:  ", any(e.kind == "clustering_embedding" for e in result.evidence))
```

```text
segments profiled: False
embedding drawn:   False
```

**The most persuasive parts of the report are the ones withheld.** A segment
profile with sizes, distinguishing features and a coloured scatter looks exactly
as convincing computed from noise as from real groups. Producing it before the
structure is established would put the most believable output on the least
supported claim, so it is gated behind two independent checks that must agree.

## The two gates

### Cluster tendency

The Hopkins statistic compares the data's own nearest-neighbour distances
against those of uniform points thrown into the same bounding box. Around 0.5,
the data is indistinguishable from noise.

```python
t = next(e for e in result.evidence if e.kind == "clustering_tendency")
print(f"{t.value['hopkins_mean']:.3f} over {t.value['repeats']} samples -> {t.value['verdict']}")
```

On the fixture this reads `0.894 ... -> clustered`; on the noise above, `0.493
... -> no_tendency`. It is repeated because a single draw varies enough to cross
a band boundary by itself.

> Hopkins drifts upward in high dimensions even on pure noise — the bounding box
> is mostly empty, so uniform points land far from everything. Read it alongside
> the distance-contrast measure rather than on its own.

### Stability, not tidiness

Silhouette, Calinski–Harabasz and Davies–Bouldin are all computed on the same
partition they judge, so they measure how *tidy* a partition is, never whether
it is real. Stability is the check that can fail: cluster two overlapping
subsamples separately, then compare them where they overlap.

```python
sweep = next(e for e in result.evidence if e.kind == "clustering_k_sweep")
for row in sweep.value["results"][:5]:
    print(f"k={row['k']} silhouette={row['silhouette']:.3f} stability={row['stability_mean']:.3f}")
print("candidate_k:", sweep.value["candidate_k"])
```

```text
k=2 silhouette=0.442 stability=1.000
k=3 silhouette=0.494 stability=1.000
k=4 silhouette=0.555 stability=1.000
k=5 silhouette=0.526 stability=0.792
k=6 silhouette=0.501 stability=0.861
candidate_k: 4
```

A candidate emerges only where both agree. On noise, k=6 has the highest
silhouette *and* respectable stability at one k — and still no candidate, because
no k clears both bars at once.

**Prism never claims a best k.** `candidate_k` is a starting point; other counts
in the sweep may be equally defensible for a different purpose.

## The segments

```python
segments = next(e for e in result.evidence if e.kind == "clustering_segments")
for s in segments.value["segments"]:
    features = "; ".join(
        f"{d['feature']} {d['direction']} ({d['z']:+.2f}σ)"
        for d in s["distinguishing_features"]
    )
    print(f"segment {s['segment']}: {s['size']} rows ({s['share']:.1%}) — {features}")
```

```text
segment 0: 207 rows (33.8%) — tenure_days lower (-1.25σ); annual_spend lower (-0.76σ); visits_per_month lower (-0.72σ)
segment 1: 182 rows (29.7%) — visits_per_year higher (+1.29σ)
segment 2: 113 rows (18.5%) — annual_spend higher (+1.99σ); tenure_days higher (+0.73σ)
segment 3: 110 rows (18.0%) — tenure_days higher (+1.32σ); visits_per_year lower (-1.20σ)
```

Groups are described in **standard deviations from the overall average**, not
raw units, so features measured in different scales are directly comparable.
Each segment also carries the real rows nearest its centre — medoids, not a
synthetic average that may resemble nobody.

## Categorical columns describe the groups; they never form them

Euclidean distance over one-hot columns asserts that every pair of categories is
exactly √2 apart, that a five-category column deserves five times the weight of
a two-category one, and that "west versus east" is commensurable with "spent 400
more". None of that is what anyone means.

So categorical columns are kept out of the distance and used for the job they
can actually do — describing the groups that were found without them:

```python
print(segments.value["uninformative_categoricals"])
```

```text
['region']
```

That is a genuine result, not a gap. `region` is random in this fixture, and the
grouping was formed with no reference to it, so "the groups do not line up with
region" is a real finding rather than a circular one. When categorical columns
*are* the signal, the report points at Gower distance or k-prototypes instead of
pretending.

## Sensitivity: choices nobody thinks of as choices

```python
sens = next(e for e in result.evidence if e.kind == "clustering_sensitivity")
print("agreement without standardizing:", sens.value["scaling_agreement"])
for drop in sens.value["feature_drops"]:
    print(f"  without {drop['feature']}: {drop['agreement_without_it']:.2f}")
```

Two questions. Standardizing or not is a default most pipelines never revisit,
and on features with very different ranges it changes the partition outright.
Dropping one feature at a time reveals whether the grouping summarises everything
or restates one column — and a one-feature grouping is a threshold on that
feature, which is worth saying plainly rather than presenting as multivariate
segmentation.

On this fixture the drop test also exposes the redundancy directly: removing
either `visits_per_month` or `visits_per_year` changes nothing, because the other
still carries the same quantity.

## What it checks

| Area | Checks |
|---|---|
| Features | Which columns build the distance, which describe it, which were excluded and why |
| Scale | Range ratio across features, and how much standardizing changes the answer |
| Redundancy | Feature pairs carrying one quantity, which a distance double-weights |
| Duplicates | Exact repeats and near-repeats that weight one region of space twice |
| Geometry | Intrinsic dimensionality via PCA, and whether distances still discriminate |
| Tendency | Repeated Hopkins statistic against uniform points in the same box |
| Search | k-sweep with silhouette, Calinski–Harabasz, Davies–Bouldin, and inertia |
| Stability | Adjusted Rand index between two independently clustered subsamples |
| Sensitivity | Scaling choice, and one-feature-out dominance |
| Segments | Sizes, distinguishing features in σ, category mix, representative rows |
| Display | A PCA projection, faceted one panel per group, labelled as a visual aid |
| Guidance | Other algorithms suited to the geometry actually measured |

## The report

```python
result.to_html("segments.html")
```

Sections: findings, alerts, **Segments** (only when structure was found), **How
many groups**, **Is it clusterable**, then the reference tables. The projection
is drawn as **small multiples — one panel per group** rather than one colour per
group, because past three groups no categorical palette can keep every pair
distinguishable for a colour-blind reader when every pair is on screen at once.
Faceting removes the question instead of losing to it.

## Limits worth knowing

- k-means drove the search because it is fast and deterministic under a fixed
  seed, not because it is the right model. Elongated, nested, or
  density-varying structure will score poorly here and still be real.
- A projection is a visual aid, never evidence. Two dimensions cannot show what
  a higher-dimensional separation looks like, so the captured-variance figure
  travels with the picture.
- Hopkins compares against uniform points in the data's own bounding box, so a
  non-rectangular but structureless region can score above 0.5 for shape alone.
- Intrinsic dimensionality is a linear notion and understates curved structure.
- Rows with missing features are median-imputed to place them in space. An
  imputed coordinate is a guess, and rows with several drift toward the centre.
- Candidate segments are descriptions of where rows fell. Clustering has no
  ground truth, so they are never a discovered category that exists in the world.

## See also

- [Anomaly detection](anomaly-detection.md) — the other unsupervised recipe
- [The baseline profile](profile.md) — feature distributions before you cluster them
- [Results & evidence](results-and-evidence.md) — the `AnalysisResult` contract
