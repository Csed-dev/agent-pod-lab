<experiment>
<description>
GNN-Ablation für PolyGCN: feste Koeffizienten c_k = 1 ohne Lernen.
Identisches Setup zu v3-gcn-capacity (PolyGCN, 4 Schichten, embed=64,
hidden=128, K=1024), aber kein Training. Die Koeffizienten werden zur
Eval-Zeit auf 1 gesetzt, was die reine Neumann-Reihe ergibt.

Da kein Training stattfindet, ist auch kein Trainings-Prior nötig.
Die Auswertung läuft direkt auf den 11 SuiteSparse-Eval-Matrizen.
</description>

<hypothesis>
Wenn dieses Experiment dieselben Scores liefert wie v3-gcn-capacity,
ist das gelernte GCN-Backbone nutzlos. v3-gcn-capacity wäre dann nur
die Reproduktion des Run-67-Resultats (PolyMPNN baseline mit K=1024,
score 0.048) bzw. der no-gnn-Ablation aus FINDINGS.md, die bereits
2026-03 dokumentiert wurde: "GNN provides ZERO benefit. Fixed c_k=1
is marginally BETTER."

Erwartung: Score sehr nah an 0.048 (FINDINGS Run 67) und SuiteSparse
Konvergenz 10/11 oder 11/11 (saylr4 hängt vom K=1024-Verhalten ab,
ohne Sign-Korrektur vermutlich 0/1 oder ähnlich).

Falls bestätigt, muss der "Durchbruch" v3-gcn-capacity-0001 ehrlich
zurückgerudert werden.
</hypothesis>

<acceptance>
- score <= 0.10
- suitesparse_conv >= 80.0
</acceptance>

<context>
Diese Ablation reproduziert für den GCN-Backbone das Vorgehen, das die
"no-gnn-0001"-Ablation für den MPNN-Backbone macht. Beide Experimente
sollen die Frage klären, ob das gelernte GNN überhaupt einen Mehrwert
über die klassische, nicht-trainierte Neumann-Reihe liefert.

Im FINDINGS.md des Thesis-Repos ist die MPNN-Variante bereits dokumentiert
(Section 2 "The Ablation That Changed Everything"): GNN provides ZERO
benefit, fixed c_k=1 is marginally BETTER. Diese Ablation für GCN ist
der notwendige Test, ob die Aussage auch für GCN-Backbones gilt.
</context>
</experiment>
