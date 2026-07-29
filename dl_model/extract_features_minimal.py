#!/usr/bin/env python3
"""
Minimal curated feature extractor — v6
Changes from v3:
  - align_longest_ungapped  : kept but NORMALIZED by alignment length (CESAR artefact noted)
  - pct_cds_retained        : kept for completeness but flagged as low-variance under CESAR
  - enc_diff                : was hardcoded 0.0, now computed properly (Wright 1990)
  - cai_ratio               : was hardcoded 1.0, now computed properly
  - orf_dominance_diff      : was hardcoded 0.0, now computed properly
  - nonsyn_spatial_entropy  : was hardcoded 0.0, now computed properly
  - period3_power           : was hardcoded 0.0, now computed properly
  - spectral_entropy        : was hardcoded 0.0, now computed properly
  - aa_freq_cosine          : NEW — cosine similarity of amino acid frequency vectors
  - d4v                     : NEW — transversion-only distance at 4-fold sites
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DNA   = set('ACGTN-')
BASES = 'ACGT'
STOPS = {'TAA', 'TAG', 'TGA'}

CODON_TABLE = {
    'ATA':'I','ATC':'I','ATT':'I','ATG':'M','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
    'AAC':'N','AAT':'N','AAA':'K','AAG':'K','AGC':'S','AGT':'S','AGA':'R','AGG':'R',
    'CTA':'L','CTC':'L','CTG':'L','CTT':'L','CCA':'P','CCC':'P','CCG':'P','CCT':'P',
    'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q','CGA':'R','CGC':'R','CGG':'R','CGT':'R',
    'GTA':'V','GTC':'V','GTG':'V','GTT':'V','GCA':'A','GCC':'A','GCG':'A','GCT':'A',
    'GAC':'D','GAT':'D','GAA':'E','GAG':'E','GGA':'G','GGC':'G','GGG':'G','GGT':'G',
    'TCA':'S','TCC':'S','TCG':'S','TCT':'S','TTC':'F','TTT':'F','TTA':'L','TTG':'L',
    'TAC':'Y','TAT':'Y','TAA':'*','TAG':'*','TGC':'C','TGT':'C','TGA':'*','TGG':'W',
}
AATOCODONS = {}
for _c, _a in CODON_TABLE.items():
    if _a != '*':
        AATOCODONS.setdefault(_a, []).append(_c)

FOURFOLD = {
    'CTT','CTC','CTA','CTG','GTT','GTC','GTA','GTG','TCT','TCC','TCA','TCG',
    'CCT','CCC','CCA','CCG','ACT','ACC','ACA','ACG','GCT','GCC','GCA','GCG',
    'CGT','CGC','CGA','CGG','GGT','GGC','GGA','GGG',
}

POLARITY = {
    'G':'nonpolar','A':'nonpolar','V':'nonpolar','L':'nonpolar','I':'nonpolar',
    'P':'nonpolar','F':'nonpolar','W':'nonpolar','M':'nonpolar',
    'S':'polar','T':'polar','Y':'polar','N':'polar','Q':'polar','C':'polar',
    'K':'positive','R':'positive','H':'positive',
    'D':'negative','E':'negative',
    '*':'stop','X':'unknown',
}

GRANTHAM = {
    ('A','R'):112,('A','N'):111,('A','D'):126,('A','C'):195,('A','Q'):91,('A','E'):107,
    ('A','G'):60,('A','H'):86,('A','I'):94,('A','K'):106,('A','L'):96,('A','M'):84,
    ('A','F'):113,('A','P'):27,('A','S'):99,('A','T'):58,('A','W'):148,('A','Y'):112,('A','V'):64,
    ('R','N'):86,('R','D'):96,('R','C'):180,('R','Q'):43,('R','E'):54,('R','G'):125,
    ('R','H'):29,('R','I'):97,('R','K'):26,('R','L'):102,('R','M'):91,('R','F'):97,
    ('R','P'):103,('R','S'):110,('R','T'):71,('R','W'):101,('R','Y'):77,('R','V'):96,
    ('N','D'):23,('N','C'):139,('N','Q'):46,('N','E'):42,('N','G'):80,('N','H'):68,
    ('N','I'):149,('N','K'):94,('N','L'):153,('N','M'):142,('N','F'):158,('N','P'):91,
    ('N','S'):46,('N','T'):65,('N','W'):174,('N','Y'):143,('N','V'):133,
    ('D','C'):154,('D','Q'):61,('D','E'):45,('D','G'):94,('D','H'):81,('D','I'):168,
    ('D','K'):101,('D','L'):172,('D','M'):160,('D','F'):177,('D','P'):108,('D','S'):65,
    ('D','T'):85,('D','W'):181,('D','Y'):160,('D','V'):152,
    ('C','Q'):154,('C','E'):170,('C','G'):159,('C','H'):174,('C','I'):198,('C','K'):202,
    ('C','L'):198,('C','M'):196,('C','F'):205,('C','P'):169,('C','S'):112,('C','T'):149,
    ('C','W'):215,('C','Y'):194,('C','V'):192,
    ('Q','E'):29,('Q','G'):87,('Q','H'):24,('Q','I'):109,('Q','K'):53,('Q','L'):113,
    ('Q','M'):101,('Q','F'):116,('Q','P'):76,('Q','S'):68,('Q','T'):42,('Q','W'):130,
    ('Q','Y'):99,('Q','V'):96,
    ('E','G'):98,('E','H'):40,('E','I'):134,('E','K'):56,('E','L'):138,('E','M'):126,
    ('E','F'):140,('E','P'):93,('E','S'):80,('E','T'):65,('E','W'):152,('E','Y'):122,('E','V'):121,
    ('G','H'):98,('G','I'):135,('G','K'):127,('G','L'):138,('G','M'):127,('G','F'):153,
    ('G','P'):42,('G','S'):56,('G','T'):59,('G','W'):184,('G','Y'):147,('G','V'):109,
    ('H','I'):94,('H','K'):32,('H','L'):99,('H','M'):87,('H','F'):100,('H','P'):77,
    ('H','S'):89,('H','T'):47,('H','W'):115,('H','Y'):83,('H','V'):84,
    ('I','K'):102,('I','L'):5,('I','M'):10,('I','F'):21,('I','P'):95,('I','S'):142,
    ('I','T'):89,('I','W'):61,('I','Y'):33,('I','V'):29,
    ('K','L'):107,('K','M'):95,('K','F'):102,('K','P'):103,('K','S'):121,('K','T'):78,
    ('K','W'):110,('K','Y'):85,('K','V'):97,
    ('L','M'):15,('L','F'):22,('L','P'):98,('L','S'):145,('L','T'):92,('L','W'):61,
    ('L','Y'):36,('L','V'):32,
    ('M','F'):28,('M','P'):87,('M','S'):135,('M','T'):81,('M','W'):67,('M','Y'):36,('M','V'):21,
    ('F','P'):114,('F','S'):155,('F','T'):103,('F','W'):40,('F','Y'):22,('F','V'):50,
    ('P','S'):74,('P','T'):38,('P','W'):147,('P','Y'):110,('P','V'):68,
    ('S','T'):58,('S','W'):177,('S','Y'):144,('S','V'):124,
    ('T','W'):128,('T','Y'):92,('T','V'):69,
    ('W','Y'):37,('W','V'):88,
    ('Y','V'):55,
}

# V6 retains every V5 feature and adds d4v, an unsaturated molecular-clock
# signal based on transversions at four-fold-degenerate third codon positions.
FEATURE_ORDER = [
    'kmer4_cosine',
    'transition_ratio',
    'cai_ratio',
    'gc_diff_overall',
    'enc_diff',
    'rscu_distance',
    'gc3_diff',
    'four_fold_id',
    'grantham_mean',
    'grantham_max',
    'orf_dominance_diff',
    'ka',
    'ks',
    'kaks_ratio',
    'kaks_window_variance',
    'pct_intact_orf',
    'nonsyn_spatial_entropy',
    'polarity_preserving_frac',
    'cpg_tpg_rate',
    'period3_power',
    'spectral_entropy',
    'exon_position_relative',
    # Kept but flagged as CESAR low-variance:
    'align_longest_ungapped_norm',
    'pct_cds_retained',
    # Additional pure functional / alignment features:
    'aa_freq_cosine',
    'n_valid_codons',
    'exon_length_ratio',
    'max_kaks_window',
    # V6: appended so every V5 feature keeps its original column index.
    'd4v',
]

# Explicit V5 contract for loading/training the existing V5 checkpoint.  New
# V6 training should use FEATURE_ORDER, which includes d4v.
FEATURE_ORDER_V5 = FEATURE_ORDER[:-1]

_AAS = sorted(set(v for v in CODON_TABLE.values() if v != '*'))

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def clean_seq(seq):
    if pd.isna(seq):
        return ''
    return ''.join(ch for ch in str(seq).upper() if ch in DNA)

def ungap(seq):
    return seq.replace('-', '')

def codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq), 3)]

def is_transition(a, b):
    return (a, b) in {('A','G'),('G','A'),('C','T'),('T','C')}

def is_fourfold_degenerate(codon):
    """Whether all third-position bases encode the same non-stop amino acid."""
    if len(codon) != 3 or '-' in codon or codon not in CODON_TABLE:
        return False
    amino_acids = {
        CODON_TABLE.get(codon[:2] + base)
        for base in BASES
    }
    return len(amino_acids) == 1 and None not in amino_acids and '*' not in amino_acids

def fourfold_transversion_distance(c1_list, c2_list):
    """K80 transversion distance at aligned four-fold-degenerate third sites.

    Only codon pairs for which *both* codons are four-fold-degenerate are used,
    which excludes gaps, ambiguous codons, and changes in the first two codon
    positions.  With Q as the fraction of these sites that differ by a
    transversion, the transversion-only K80 correction is -0.5 ln(1 - 2Q).
    The same finite saturation cap used for Ka/Ks keeps downstream scaling
    stable for the mathematically uncorrectable Q >= 0.5 case.
    """
    sites = transversions = 0
    for cod1, cod2 in zip(c1_list, c2_list):
        if not (is_fourfold_degenerate(cod1) and is_fourfold_degenerate(cod2)):
            continue
        sites += 1
        if cod1[2] != cod2[2] and not is_transition(cod1[2], cod2[2]):
            transversions += 1

    if sites == 0:
        return np.nan
    q = transversions / sites
    return -0.5 * math.log(1 - 2 * q) if q < 0.5 else 10.0

def grantham_dist(a1, a2):
    if a1 == a2 or a1 in ('*','X') or a2 in ('*','X'):
        return 0
    return GRANTHAM.get((a1, a2), GRANTHAM.get((a2, a1), 0))

def gc_content(seq):
    s = ungap(seq)
    if not s:
        return np.nan
    return (s.count('G') + s.count('C')) / len(s)

# ---------------------------------------------------------------------------
# k-mer cosine
# ---------------------------------------------------------------------------
def kmer_cosine(s1, s2, k=4):
    s1, s2 = ungap(s1), ungap(s2)
    c1 = Counter(s1[i:i+k] for i in range(max(0, len(s1)-k+1)))
    c2 = Counter(s2[i:i+k] for i in range(max(0, len(s2)-k+1)))
    keys = set(c1) | set(c2)
    if not keys:
        return np.nan
    v1 = np.array([c1.get(km, 0) for km in keys], dtype=float)
    v2 = np.array([c2.get(km, 0) for km in keys], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else np.nan

# ---------------------------------------------------------------------------
# Amino acid frequency cosine  (NEW — pure functional identity signal)
# ---------------------------------------------------------------------------
def aa_freq_cosine(c1_list, c2_list):
    """
    Cosine similarity of 20-dim amino acid frequency vectors.
    Captures functional class identity — orthologs share ancestral AA composition
    shaped by the same structural/catalytic constraints.  Paralogs from different
    gene families have systematically different AA compositions.
    Completely insensitive to whether the gene is intact or degraded.
    """
    def aa_vec(clist):
        counts = Counter(
            CODON_TABLE[c] for c in clist
            if c in CODON_TABLE and CODON_TABLE[c] != '*' and '-' not in c
        )
        total = sum(counts.values())
        if total == 0:
            return None
        return np.array([counts.get(aa, 0) / total for aa in _AAS], dtype=float)

    v1 = aa_vec(c1_list)
    v2 = aa_vec(c2_list)
    if v1 is None or v2 is None:
        return np.nan
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else np.nan

# ---------------------------------------------------------------------------
# Ka/Ks (Nei-Gojobori)
# ---------------------------------------------------------------------------
def codon_degeneracy(codon):
    if len(codon) != 3 or '-' in codon or codon not in CODON_TABLE:
        return 0.0, 0.0
    aa = CODON_TABLE[codon]
    ssites = 0.0
    for pos in range(3):
        syn = tot = 0
        for b in BASES:
            if b == codon[pos]:
                continue
            m = codon[:pos] + b + codon[pos+1:]
            if m in CODON_TABLE:
                tot += 1
                if CODON_TABLE[m] == aa:
                    syn += 1
        if tot:
            ssites += syn / tot
    return ssites, 3.0 - ssites

def nei_gojobori_kaks(c1_list, c2_list):
    total_s = total_n = sd = nd = 0.0
    for c1, c2 in zip(c1_list, c2_list):
        if len(c1) != 3 or len(c2) != 3 or '-' in c1 or '-' in c2:
            continue
        if c1 not in CODON_TABLE or c2 not in CODON_TABLE:
            continue
        s1, n1 = codon_degeneracy(c1)
        s2, n2 = codon_degeneracy(c2)
        total_s += (s1 + s2) / 2.0
        total_n += (n1 + n2) / 2.0
        if c1 != c2:
            if CODON_TABLE[c1] == CODON_TABLE[c2]:
                sd += 1
            else:
                nd += 1
    if total_s == 0 or total_n == 0:
        return np.nan, np.nan, np.nan
    p_s, p_n = sd / total_s, nd / total_n
    ks = -0.75 * math.log(1 - 4*p_s/3) if p_s < 0.75 else 10.0
    ka = -0.75 * math.log(1 - 4*p_n/3) if p_n < 0.75 else 10.0
    return ka, ks, ka/ks if ks > 0 else np.nan

def sliding_kaks_stats(c1_list, c2_list, window=30, step=10):
    if len(c1_list) < window:
        return np.nan, np.nan
    vals = []
    for i in range(0, len(c1_list) - window + 1, step):
        _, _, r = nei_gojobori_kaks(c1_list[i:i+window], c2_list[i:i+window])
        if not (r is None or np.isnan(r)):
            vals.append(r)
    if not vals:
        return np.nan, np.nan
    var = float(np.var(vals)) if len(vals) >= 2 else 0.0
    mmax = float(np.max(vals))
    return var, mmax

# ---------------------------------------------------------------------------
# RSCU distance
# ---------------------------------------------------------------------------
def rscu_vec(clist):
    counts = Counter(c for c in clist if c in CODON_TABLE and CODON_TABLE[c] != '*')
    out = {}
    for aa, syn in AATOCODONS.items():
        total = sum(counts.get(c, 0) for c in syn)
        exp = total / len(syn) if syn else 0
        for c in syn:
            out[c] = counts.get(c, 0) / max(exp, 1)
    return out

def rscu_distance(c1_list, c2_list):
    r1, r2 = rscu_vec(c1_list), rscu_vec(c2_list)
    keys = sorted(set(r1) | set(r2))
    return float(math.sqrt(sum((r1.get(k, 0) - r2.get(k, 0))**2 for k in keys)))

# ---------------------------------------------------------------------------
# CAI  (human reference weights)
# ---------------------------------------------------------------------------
_CAI_W = {
    'GCA':0.58,'GCC':1.00,'GCG':0.26,'GCT':0.68,'TGC':1.00,'TGT':0.47,
    'GAC':1.00,'GAT':0.43,'GAA':0.44,'GAG':1.00,'TTC':1.00,'TTT':0.43,
    'GGA':0.44,'GGC':1.00,'GGG':0.43,'GGT':0.35,'CAC':1.00,'CAT':0.41,
    'ATA':0.16,'ATC':1.00,'ATT':0.36,'AAA':0.43,'AAG':1.00,
    'CTA':0.18,'CTC':1.00,'CTG':1.00,'CTT':0.33,'TTA':0.19,'TTG':0.33,
    'ATG':1.00,'AAC':1.00,'AAT':0.44,'CCA':0.56,'CCC':1.00,'CCG':0.22,'CCT':0.53,
    'CAA':0.26,'CAG':1.00,'AGA':0.21,'AGG':0.23,'CGA':0.11,'CGC':1.00,'CGG':0.21,'CGT':0.09,
    'AGC':1.00,'AGT':0.32,'TCA':0.29,'TCC':0.74,'TCG':0.14,'TCT':0.44,
    'ACA':0.55,'ACC':1.00,'ACG':0.17,'ACT':0.52,'GTA':0.26,'GTC':1.00,'GTG':1.00,'GTT':0.28,
    'TGG':1.00,'TAC':1.00,'TAT':0.40,
}

def cai(clist):
    ws = [_CAI_W[c] for c in clist if c in _CAI_W]
    if not ws:
        return np.nan
    return float(math.exp(sum(math.log(w) for w in ws if w > 0) / len(ws)))

# ---------------------------------------------------------------------------
# ENC  (Wright 1990) — properly computed
# ---------------------------------------------------------------------------
def enc(clist):
    counts = Counter(c for c in clist if c in CODON_TABLE and CODON_TABLE[c] != '*')
    family_f = {}
    for aa, syn in AATOCODONS.items():
        nsyn = len(syn)
        if nsyn <= 1:
            continue
        total = sum(counts.get(c, 0) for c in syn)
        if total <= 1:
            continue
        pi_sq = sum((counts.get(c, 0)/total)**2 for c in syn)
        f = (total * pi_sq - 1) / (total - 1)
        family_f.setdefault(nsyn, []).append(f)
    if not family_f:
        return np.nan
    enc_val = 0.0
    for nsyn, vals in family_f.items():
        mean_f = float(np.mean(vals))
        if mean_f > 0:
            enc_val += nsyn / mean_f
    return float(min(enc_val, 61.0))

# ---------------------------------------------------------------------------
# ORF dominance
# ---------------------------------------------------------------------------
def orf_dominance(seq):
    s = ungap(seq)
    if len(s) < 3:
        return 0.0
    best = 0
    for frame in range(3):
        run = best_f = 0
        for i in range(frame, len(s)-2, 3):
            cod = s[i:i+3]
            if cod in STOPS:
                run = 0
            else:
                run += 1
                best_f = max(best_f, run)
        best = max(best, best_f)
    return best / max(1, len(s) // 3)

# ---------------------------------------------------------------------------
# Periodicity / spectral
# ---------------------------------------------------------------------------
def substitution_periodicity(seq1, seq2):
    signal = np.array(
        [1 if a != b else 0 for a, b in zip(seq1, seq2) if a != '-' and b != '-'],
        dtype=float
    )
    if len(signal) < 12:
        return np.nan, np.nan
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))
    idx = int(np.argmin(np.abs(freqs - 1/3)))
    total = fft.sum() + 1e-10
    p3 = float(fft[idx] / total)
    psd = fft / total
    ent = float(-np.sum(psd * np.log(psd + 1e-10)))
    return p3, ent

# ---------------------------------------------------------------------------
# Spatial entropy of nonsynonymous substitutions
# ---------------------------------------------------------------------------
def nonsyn_entropy(positions, total_len):
    if not positions or total_len <= 0:
        return 0.0
    counts, _ = np.histogram(positions, bins=10, range=(0, total_len))
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------
def extract_features(seq1, seq2, exon_number=None, total_exons=None):
    seq1 = clean_seq(seq1)
    seq2 = clean_seq(seq2)
    c1 = codons(seq1)
    c2 = codons(seq2)
    L = min(len(c1), len(c2))
    c1, c2 = c1[:L], c2[:L]

    if L == 0:
        return {k: np.nan for k in FEATURE_ORDER}

    transitions = transversions = 0
    nonsyn_pos = []
    g_scores = []
    pol_pres = pol_tot = 0
    ff_tot = ff_match = 0
    valid_gc3_1 = valid_gc3_2 = gc3_1 = gc3_2 = 0
    cpg_tot = cpg_tpg = 0
    intact_stretches = []
    cur_stretch = 0

    for i in range(min(len(seq1), len(seq2)) - 1):
        if seq1[i:i+2] == 'CG':
            cpg_tot += 1
            if seq2[i:i+2] in ('TG', 'CA'):
                cpg_tpg += 1

    for i, (cod1, cod2) in enumerate(zip(c1, c2)):
        if len(cod1) != 3 or len(cod2) != 3:
            continue
        if cod1 in FOURFOLD and '-' not in cod1 and '-' not in cod2:
            ff_tot += 1
            if cod1[2] == cod2[2]:
                ff_match += 1
        if '-' not in cod1:
            valid_gc3_1 += 1
            gc3_1 += int(cod1[2] in 'GC')
        if '-' not in cod2:
            valid_gc3_2 += 1
            gc3_2 += int(cod2[2] in 'GC')
        if cod2 in STOPS or ('-' in cod2 and cod2.count('-') % 3 != 0):
            intact_stretches.append(cur_stretch)
            cur_stretch = 0
        else:
            cur_stretch += 1
        if '-' not in cod1 and '-' not in cod2 and cod1 in CODON_TABLE and cod2 in CODON_TABLE:
            aa1, aa2 = CODON_TABLE[cod1], CODON_TABLE[cod2]
            for b1, b2 in zip(cod1, cod2):
                if b1 != b2:
                    if is_transition(b1, b2):
                        transitions += 1
                    else:
                        transversions += 1
            if aa1 != aa2:
                nonsyn_pos.append(i)
                g_scores.append(grantham_dist(aa1, aa2))
                p1, p2 = POLARITY.get(aa1), POLARITY.get(aa2)
                if p1 == p2 and p1 not in (None, 'unknown', 'stop'):
                    pol_pres += 1
                pol_tot += 1
    intact_stretches.append(cur_stretch)

    ka, ks, kaks = nei_gojobori_kaks(c1, c2)
    kaks_var, kaks_max = sliding_kaks_stats(c1, c2)
    d4v = fourfold_transversion_distance(c1, c2)
    cai1, cai2 = cai(c1), cai(c2)
    enc1, enc2 = enc(c1), enc(c2)
    od1, od2   = orf_dominance(seq1), orf_dominance(seq2)
    p3, sent   = substitution_periodicity(seq1, seq2)

    # align_longest_ungapped normalized by alignment length (CESAR artefact aware)
    paired = ''.join('x' if a != '-' and b != '-' else ' ' for a, b in zip(seq1, seq2))
    longest_run = max((len(s) for s in paired.split()), default=0)
    align_len   = min(len(seq1), len(seq2))

    ungap1 = len(ungap(seq1))
    ungap2 = len(ungap(seq2))

    # Number of aligned codon pairs usable by the v4 RNN.  This is part of
    # the feature contract used when dl_orthology_model_v4.pth was trained.
    n_valid = sum(1 for c1, c2 in zip(c1, c2) if '-' not in c1 and '-' not in c2)

    return {
        'kmer4_cosine':              kmer_cosine(seq1, seq2, 4),
        'transition_ratio':          transitions / max(1, transitions + transversions),
        'cai_ratio':                 cai2 / max(0.001, cai1) if not (np.isnan(cai1) or np.isnan(cai2)) else np.nan,
        'gc_diff_overall':           abs(gc_content(seq1) - gc_content(seq2)) if not np.isnan(gc_content(seq1)) else np.nan,
        'enc_diff':                  abs(enc1 - enc2) if not (np.isnan(enc1) or np.isnan(enc2)) else np.nan,
        'rscu_distance':             rscu_distance(c1, c2),
        'gc3_diff':                  abs(gc3_1/max(1, valid_gc3_1) - gc3_2/max(1, valid_gc3_2)),
        'four_fold_id':              ff_match / max(1, ff_tot),
        'd4v':                       d4v,
        'grantham_mean':             float(np.mean(g_scores)) if g_scores else 0.0,
        'grantham_max':              float(np.max(g_scores))  if g_scores else 0.0,
        'orf_dominance_diff':        abs(od1 - od2),
        'ka':                        ka,
        'ks':                        ks,
        'kaks_ratio':                kaks,
        'kaks_window_variance':      kaks_var,
        'pct_intact_orf':            max(intact_stretches) / max(1, L),
        'nonsyn_spatial_entropy':    nonsyn_entropy(nonsyn_pos, L),
        'polarity_preserving_frac':  pol_pres / pol_tot if pol_tot > 0 else 1.0,
        'cpg_tpg_rate':              cpg_tpg / max(1, cpg_tot),
        'period3_power':             p3   if p3   is not None else np.nan,
        'spectral_entropy':          sent if sent is not None else np.nan,
        'exon_position_relative':    exon_number / total_exons if (exon_number is not None and total_exons not in (None, 0)) else np.nan,
        # CESAR low-variance — kept for completeness, flagged
        'align_longest_ungapped_norm': longest_run / align_len if align_len > 0 else np.nan,
        'pct_cds_retained':          ungap2 / max(1, ungap1),
        # NEW — pure functional class identity
        'aa_freq_cosine':            aa_freq_cosine(c1, c2),
        'n_valid_codons':            n_valid,
        'exon_length_ratio':         ungap1 / max(1, ungap2),
        'max_kaks_window':           kaks_max,
    }

# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------
def infer_total_exons(df):
    if 'exon_number' in df.columns and 'transcript_id' in df.columns:
        return df.groupby('transcript_id')['exon_number'].transform('max')
    return pd.Series([np.nan] * len(df))

def process_file_to_npz(infile, outfile):
    print(f'Reading {infile} ...')
    df = pd.read_csv(infile)
    for col in ('ref_seq', 'query_seq', 'label'):
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    if 'total_exons' not in df.columns:
        df['total_exons'] = infer_total_exons(df)

    X_rows, y_list, groups = [], [], []
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        feats = extract_features(
            row.ref_seq, row.query_seq,
            getattr(row, 'exon_number', None),
            getattr(row, 'total_exons', None),
        )
        X_rows.append([feats[k] for k in FEATURE_ORDER])
        y_list.append(row.label)
        groups.append(getattr(row, 'transcript_id', idx))
        if idx % 10000 == 0:
            print(f'  {idx}/{total}')

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    np.savez_compressed(
        outfile,
        X=X, y=y,
        features=np.array(FEATURE_ORDER, dtype=object),
        groups=np.array(groups, dtype=object),
    )
    print(f'Saved {outfile}  shape={X.shape}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='train_cleaned.csv')
    p.add_argument('--test',  default='test_cleaned.csv')
    p.add_argument('--outdir', default='.')
    args = p.parse_args()
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    process_file_to_npz(args.train, od / 'train.npz')
    if Path(args.test).exists():
        process_file_to_npz(args.test, od / 'test.npz')

if __name__ == '__main__':
    main()
