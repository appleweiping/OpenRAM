# OpenRAM leakage diagnostic, 2026-11-06

This archives the actual full-table experiment for issue333, not a proposed
production model. The single predeclared log-target candidate failed its frozen
acceptance criteria; it was not tuned or installed in the compiler.

Base data/code: dev d93bdaa3c61530e1ed9e9f38a8f7904793f71795. Both complete
shipped tables are used (324 and360 rows), all10 out-of-fold CSVs are included,
and CSV/source/protocol hashes and dependency versions are retained.
The reference arm is actual OpenRAM Ridge on the existing globally scaled data.
The controlled arms both fit feature MinMaxScaler only on the training fold,
then Ridge(alpha=1); the candidate alone transforms the positive raw target with
log/exp. Ten folds use shuffle=False. Geometry holdout uses exactly the first
four structural inputs; no output/sample identifier is a grouping feature.

Read leakage-evaluation-protocol.json first: it was frozen before comparing
the candidates. leakage-comparison.json contains all fold metrics, held-out
indices, aggregate metrics and frozen-gate decisions. Each *-oof.csv contains
every row's index, fold, original target and prediction. geometry-target-audit
records the nominal PVT coverage and constant-target groups. Within-geometry
R2 is not a useful quality summary for constant targets (floating roundoff can
produce huge values); the predeclared grouped quality gate uses pooled OOF R2.
Pooled and fold-mean R2 are intentionally both retained for the10fold results.

The runner/config/shell script are exact archival copies of the actual run.
They retain its explicit E:/mnt/e paths and existing read-only Python venv;
they are not a portable environment installer. To reproduce elsewhere, adapt
only those location/environment paths, with a normal initialized OpenRAM
environment and the versions recorded in the JSON. No weights, PDK models or
simulator-generated data are included or fabricated. All input CSVs remain
available at the pinned upstream commit.

The independent CV-preprocessing correction is PR341; this diagnostic branch
also contains that commit but the artifacts are intentionally outside the PR.
It changes evaluation only, leaving all6156 actual production values bitwise
unchanged. This does not certify leakage physics, PVT generalization, SPICE,
a new research method or SOTA performance. No Liberty goldens were changed.

Assisted-by: OpenAI Codex.
