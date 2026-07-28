"""Vérificateur d'artefact : teste des assertions sur le PDF COMPILÉ, pas sur le .tex.

Raison d'être : un `str.replace` qui ne trouve pas sa cible échoue en silence, et un rapport de
correction peut donc être faux. On ne vérifie que l'objet final.

Deux pièges rencontrés, tous deux traités ici :
  - l'extraction PDF coupe les phrases par des sauts de ligne  -> normaliser les blancs
  - LaTeX césure les mots en fin de ligne ("indepen-\\ndently") -> retirer les césures
Usage : python verify.py [paper.pdf]
"""
from __future__ import annotations
import re, sys
import pypdf


def _canon(s):
    """Normalisation commune au texte extrait et aux motifs.

    Trois pièges rencontrés, tous traités ici :
      - l'extraction PDF coupe les phrases par des sauts de ligne  -> blancs normalisés
      - LaTeX césure en fin de ligne ("indepen-\\ndently")          -> césures retirées
      - LaTeX insère une espace fine après les virgules en mode math -> ", " -> ","
      - idem après "(" et "[" et avant ")" et "]"
    Sans ces quatre, on produit de FAUX échecs (constaté quatre fois dans ce projet).
    """
    s = re.sub(r"-\s*\n\s*", "", s)          # césure de fin de ligne
    s = re.sub(r"\s+", " ", s)                 # sauts de ligne
    s = re.sub(r",\s*", ",", s)                # espace fine après virgule (mode math)
    s = re.sub(r"([(\[])\s+", r"\1", s)        # espace après ( ou [
    return re.sub(r"\s+([)\]])", r"\1", s)     # espace avant ) ou ]


def text_of(path):
    raw = "".join((p.extract_text() or "") for p in pypdf.PdfReader(path).pages)
    return _canon(raw)


def norm(s):
    return _canon(s)


# (libellé, motif, doit_être_présent)
CHECKS = [
    ("titre courant",                  "How fine a change can moments see?", True),
    ("ancien titre absent",            "When is persistent homology redundant", False),
    ("intro ouvre sur la détection",   "Deciding that a stream of high-dimensional embeddings has changed", True),
    ("Fig.5 : retrait explicite",      "we withdraw it as a statement about persistent homology", True),
    ("Fig.5 : ancienne lecture absente", "Delta top degrades with ambient", False),
    ("§6.9 desambiguisé",              "landscape interval on the covariance-preserving attack", True),
    ("§6.9 : plus d''adaptive attack'","landscape interval on the adaptive attack", False),
    ("citations : TwoNN",              "(TwoNN; [", True),
    ("citations : LID",                "(related to LID; [", True),
    ("citations : Fasy",               "confidence bands as in [", True),
    ("Hamburger non doublé",           "moments, Hamburger)", False),
    ("note inter-runs",                "Tables produced by different experiments re-draw their windows independently", True),
    ("Table 4 AG News complète",       "AG News (press) 0.81 [0.67, 0.94] 1.00 [1.00, 1.00]", True),
    ("récit périmé absent (1)",        "uniformly negative", False),
    ("récit périmé absent (2)",        "recalls 0.00 against", False),
    ("sigma: protocole unique",        "applied it uniformly to three settings", True),
    ("sigma: 3 réglages nommés",       "bge-1024 / AG News", True),
    ("sigma: répétitions déclarées",   "with three repetitions per cell", True),
    ("sigma: pas de constante serrée", "we do not claim a sharp constant", True),
    ("sigma: plus de 'median 1.08'",   "median 1.08", False),
    ("adversaire MMD-aware present",   "An MMD-aware adversary", True),
    ("adversaire MMD-aware: caveat",   "not an impossibility", True),
    ("KS-radial: cout rempli",         "KS-radial 1.00 1.00 0.15 4", True),
    ("KS-radial: discute",             "A bandwidth-free competitor", True),
    ("plancher: testé",                "Is there a resolution floor?", True),
    ("plancher: réfuté",               "no evidence of a floor down to", True),
    ("plancher: monotonie retirée",    "not monotone, so the earlier trend was run-to-run variation", True),
    ("plancher: plus de 'leave to future work'", "requires a fourth, smaller scale, which we leave to future work", False),
    ("sigma canonique: mediane 1.12",  "median is 1.12 with interquartile range [1.01,1.52]", True),
    ("sigma: argmax mauvais estimateur","the global argmax is the wrong estimator for this quantity", True),
    ("sigma: historique des chiffres", "in earlier versions of this work (0.48, then 1.05, then 1.64", True),
    ("sigma: plus de mediane 1.64",    "median 1.64", False),
    ("sigma: plus de 'range [1.09,2.55]'", "range [1.09,2.55]", False),
    ("conclusion propagee",            "median 1.12, IQR [1.01,1.52] over three settings", True),
    ("boucle: compensation recalibree","partial compensation of two errors", True),
    ("boucle: limite de portee",       "we have not repeated the closure on the other two settings", True),
    ("bimodalite: constatee",          "curve is multimodal", True),
    ("bimodalite: pic a l'echelle nuage","1.01 times the median pairwise distance", True),
    ("bimodalite: pic en eps partout", "is present in every repetition", True),
    ("n=26 explique",                  "produced no interior optimum at any mass fraction in the grid and is excluded", True),
    ("orpheline corrigee",             "An earlier single-run sweep produced ratios", True),
    ("plus de '0.97,1.08,1.31' orphelin","first three ratios happened to increase", False),
    ("affiliation remplie",            "Independent researcher", True),
    ("contact rempli",                 "kaleche@gmail.com", True),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper.pdf"
    T = text_of(path)
    ok = True
    for label, pat, want in CHECKS:
        found = norm(pat) in T
        good = (found == want)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label}")
    ph = sorted(set(re.findall(r"\[REPOSITORY URL\]", T)))
    print(f"\n  {'PASS' if not ph else 'TODO'}  placeholders : {ph if ph else 'aucun'}")
    print("\nARTEFACT :", "conforme" if ok else "NON CONFORME")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
