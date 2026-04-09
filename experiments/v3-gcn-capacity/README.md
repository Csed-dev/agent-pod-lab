<experiment>
<description>
v3-GCN mit hoher Kapazität und vollem Trainings-Prior auf SuiteSparse.
Architektur: PolyGCN aus ADR-13 (4 Schichten, embed=64, hidden=128, K=1024).
Etwa 140k Parameter, also rund 32x mehr als die CPU-Variante mit 4320 Parametern.
Identische Trainings-Pipeline wie baseline (PolyMPNN K=1024).
8 Trainings-Domänen, vollständiger Prior wie im Hauptbenchmark.
</description>

<hypothesis>
Die Nachtschicht 2026-04-09 hat acht v3-Experimente durchgeführt (siehe
docs/research/v3-experiment im Thesis-Repo). Die kompakte Variante
(4320 Parameter, K=64, schmaler Prior) scheiterte als universeller Preconditioner
auf SuiteSparse: Friedman-Rang 1 ging an die feste Neumann-Reihe (c_k=1),
alle paarweisen Wilcoxon-Tests p=1.0, auf thermal sogar 16x schlechter als fest.

Diese Ablation testet die offene Frage: Ist die Schwäche parametrisch oder
strukturell? Mit 32x mehr Parametern, K=1024 statt K=64, vollem 8-Domänen-Prior
und identischer Trainings-Infrastruktur wie baseline (PolyMPNN) sollte die
parametrische Hypothese geprüft werden.

Erwartung: Falls die Limitation strukturell ist (lokale GCN-Aggregation,
Polynom-Kopf), wird auch die große Variante auf SuiteSparse nicht signifikant
besser sein als die feste Neumann-Reihe und insbesondere auf thermal weiter
versagen. Falls die Limitation parametrisch ist, sollte sie sich erholen.
</hypothesis>

<acceptance>
- score <= 0.15
- suitesparse_conv >= 70.0
</acceptance>

<context>
Diese Variante verwendet GCN-Backbone (Kipf-Welling mit Gershgorin-Normalisierung)
statt MPNN. Im Gegensatz zu PolyMPNN gibt es keine Edge-Features. Die Aggregation
ist sparse_mm(AA_normalized, H), gefolgt von linearer Projektion und Skip-Connection.

Die Architektur und alle Hilfsfunktionen liegen in lib/architectures/poly_gcn.py
als direkter Spiegel zu lib/architectures/neumann.py.

Dieses Experiment ist Teil des MatrixPFN-Bachelor-Thesis-Projekts. Siehe ADR-13
im Thesis-Repo (docs/adr/13-v3-minimal-reference-architecture.md) und
docs/organisation/handouts/v3-overview.pdf für die vollständige Vorgeschichte.
</context>
</experiment>
