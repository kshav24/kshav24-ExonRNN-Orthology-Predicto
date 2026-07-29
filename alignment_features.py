import math
import numpy as np
import zlib
from collections import Counter
from Bio.Data import CodonTable

standard_table = CodonTable.unambiguous_dna_by_name["Standard"]

# valid coding codons (exclude stops)
CODONS = [c for c in standard_table.forward_table]
AA_MAP = standard_table.forward_table
# Add stop codons to AA_MAP for easier translation
AA_MAP['TAA'] = '*'
AA_MAP['TAG'] = '*'
AA_MAP['TGA'] = '*'

# --- Amino acid properties ---
hydrophobicity = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
    'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
    'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3, '*':0.0, 'X':0.0
}

charge = {
    'D':-1,'E':-1,'K':1,'R':1,'H':0.5, '*':0.0, 'X':0.0
}

# Condensed Grantham Matrix (upper triangular approximation for valid AAs)
grantham_matrix = {
    'A': {'A':0,'R':112,'N':111,'D':126,'C':195,'Q':91,'E':107,'G':60,'H':86,'I':94,'L':96,'K':106,'M':84,'F':113,'P':27,'S':99,'T':58,'W':148,'Y':112,'V':64},
    'R': {'R':0,'N':86,'D':96,'C':180,'Q':43,'E':54,'G':125,'H':29,'I':97,'L':102,'K':26,'M':91,'F':97,'P':103,'S':110,'T':71,'W':101,'Y':77,'V':96},
    'N': {'N':0,'D':23,'C':139,'Q':46,'E':42,'G':80,'H':68,'I':149,'L':153,'K':94,'M':142,'F':158,'P':91,'S':46,'T':65,'W':174,'Y':143,'V':133},
    'D': {'D':0,'C':154,'Q':61,'E':45,'G':94,'H':81,'I':168,'L':172,'K':101,'M':160,'F':177,'P':108,'S':65,'T':85,'W':181,'Y':160,'V':152},
    'C': {'C':0,'Q':154,'E':170,'G':159,'H':174,'I':198,'L':198,'K':202,'M':196,'F':205,'P':169,'S':112,'T':149,'W':215,'Y':194,'V':192},
    'Q': {'Q':0,'E':29,'G':87,'H':24,'I':109,'L':113,'K':53,'M':101,'F':116,'P':76,'S':68,'T':42,'W':130,'Y':99,'V':96},
    'E': {'E':0,'G':98,'H':40,'I':134,'L':138,'K':56,'M':126,'F':140,'P':93,'S':80,'T':65,'W':152,'Y':122,'V':121},
    'G': {'G':0,'H':98,'I':135,'L':138,'K':127,'M':127,'F':153,'P':42,'S':56,'T':59,'W':184,'Y':147,'V':109},
    'H': {'H':0,'I':94,'L':99,'K':32,'M':87,'F':100,'P':77,'S':89,'T':47,'W':115,'Y':83,'V':84},
    'I': {'I':0,'L':5,'K':102,'M':10,'F':21,'P':95,'S':142,'T':89,'W':61,'Y':33,'V':29},
    'L': {'L':0,'K':107,'M':15,'F':22,'P':98,'S':145,'T':92,'W':61,'Y':36,'V':32},
    'K': {'K':0,'M':95,'F':102,'P':103,'S':121,'T':78,'W':110,'Y':85,'V':97},
    'M': {'M':0,'F':28,'P':87,'S':135,'T':81,'W':67,'Y':36,'V':21},
    'F': {'F':0,'P':114,'S':155,'T':103,'W':40,'Y':22,'V':50},
    'P': {'P':0,'S':74,'T':38,'W':147,'Y':110,'V':68},
    'S': {'S':0,'T':58,'W':177,'Y':144,'V':124},
    'T': {'T':0,'W':128,'Y':92,'V':69},
    'W': {'W':0,'Y':37,'V':88},
    'Y': {'Y':0,'V':55},
    'V': {'V':0}
}

def get_grantham(aa1, aa2):
    if aa1 == aa2: return 0
    if aa1 in ('*', 'X') or aa2 in ('*', 'X'): return 0
    if aa1 in grantham_matrix and aa2 in grantham_matrix[aa1]:
        return grantham_matrix[aa1][aa2]
    if aa2 in grantham_matrix and aa1 in grantham_matrix[aa2]:
        return grantham_matrix[aa2][aa1]
    return 0

def aa_charge(a):
    return charge.get(a, 0)

# 4-fold degenerate codons (amino acid does not change if 3rd pos changes)
FOUR_FOLD_CODONS = {
    'CTT','CTC','CTA','CTG', # Leu
    'GTT','GTC','GTA','GTG', # Val
    'TCT','TCC','TCA','TCG', # Ser
    'CCT','CCC','CCA','CCG', # Pro
    'ACT','ACC','ACA','ACG', # Thr
    'GCT','GCC','GCA','GCG', # Ala
    'CGT','CGC','CGA','CGG', # Arg
    'GGT','GGC','GGA','GGG'  # Gly
}

def is_transition(a, b):
    return (a+b in ["AG","GA","CT","TC"])

def classify_substitution(c1, c2):
    if "-" in c1 or "-" in c2: return None
    if c1 not in AA_MAP or c2 not in AA_MAP: return None

    aa1 = AA_MAP[c1]
    aa2 = AA_MAP[c2]
    synonymous = (aa1 == aa2)

    transitions = transversions = 0
    for n1, n2 in zip(c1, c2):
        if n1 != n2:
            if is_transition(n1, n2): transitions += 1
            else: transversions += 1

    radical = abs(hydrophobicity.get(aa1, 0) - hydrophobicity.get(aa2, 0)) > 2.5
    return synonymous, transitions, transversions, radical, aa1, aa2

def codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq), 3)]

def shannon_entropy(positions, total_len):
    if not positions or total_len == 0: return 0
    # Create normalized histogram
    hist_bins = 10
    counts, _ = np.histogram(positions, bins=hist_bins, range=(0, total_len))
    probs = counts[counts > 0] / len(positions)
    return -np.sum(probs * np.log2(probs))

def calculate_compression_ratio(seq1, seq2):
    # compression distance metric
    c1 = len(zlib.compress(seq1.encode('utf-8')))
    c2 = len(zlib.compress(seq2.encode('utf-8')))
    c12 = len(zlib.compress((seq1 + seq2).encode('utf-8')))
    if c1 + c2 == 0: return 0
    return c12 / (c1 + c2)

# --- BLOSUM62 Substitution Matrix ---
_BLOSUM62_AA = 'ARNDCQEGHILKMFPSTWYV'
_BLOSUM62_DATA = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]
]
BLOSUM62 = {}
for i, a1 in enumerate(_BLOSUM62_AA):
    for j, a2 in enumerate(_BLOSUM62_AA):
        BLOSUM62[(a1, a2)] = _BLOSUM62_DATA[i][j]

def blosum62_score(aa1, aa2):
    """Look up BLOSUM62 score for an amino acid pair."""
    if aa1 in ('*', 'X') or aa2 in ('*', 'X'):
        return -4  # penalty for unknown/stop
    return BLOSUM62.get((aa1, aa2), -4)


# --- Nei-Gojobori Ka/Ks ---
def _codon_degeneracy(codon):
    """Count synonymous and non-synonymous sites for a single codon.
    Each position contributes a fraction to S and N based on how many
    single-nucleotide changes at that position are synonymous."""
    bases = 'ACGT'
    if '-' in codon or len(codon) != 3:
        return 0, 0
    original_aa = AA_MAP.get(codon)
    if original_aa is None:
        return 0, 0
    s_sites = 0.0
    for pos in range(3):
        syn_changes = 0
        total_changes = 0
        for b in bases:
            if b == codon[pos]:
                continue
            mutant = codon[:pos] + b + codon[pos+1:]
            mutant_aa = AA_MAP.get(mutant)
            if mutant_aa is None:
                continue
            total_changes += 1
            if mutant_aa == original_aa:
                syn_changes += 1
        if total_changes > 0:
            s_sites += syn_changes / total_changes
    n_sites = 3.0 - s_sites
    return s_sites, n_sites

def nei_gojobori_kaks(codon_list1, codon_list2):
    """Compute Ka and Ks using the Nei-Gojobori (1986) method with
    Jukes-Cantor correction. Returns (Ka, Ks, Ka/Ks ratio)."""
    total_S = 0.0  # synonymous sites
    total_N = 0.0  # non-synonymous sites
    sd = 0  # synonymous differences
    nd = 0  # non-synonymous differences

    for c1, c2 in zip(codon_list1, codon_list2):
        if '-' in c1 or '-' in c2 or len(c1) != 3 or len(c2) != 3:
            continue
        if c1 not in AA_MAP or c2 not in AA_MAP:
            continue

        s1, n1 = _codon_degeneracy(c1)
        s2, n2 = _codon_degeneracy(c2)
        total_S += (s1 + s2) / 2.0
        total_N += (n1 + n2) / 2.0

        aa1 = AA_MAP[c1]
        aa2 = AA_MAP[c2]
        if c1 != c2:
            if aa1 == aa2:
                sd += 1
            else:
                nd += 1

    if total_S == 0 or total_N == 0:
        return np.nan, np.nan, np.nan

    pS = sd / total_S
    pN = nd / total_N

    # Jukes-Cantor correction: d = -3/4 * ln(1 - 4/3 * p)
    try:
        Ks = -0.75 * math.log(1.0 - (4.0/3.0) * pS) if pS < 0.75 else 10.0
    except (ValueError, ZeroDivisionError):
        Ks = 10.0
    try:
        Ka = -0.75 * math.log(1.0 - (4.0/3.0) * pN) if pN < 0.75 else 10.0
    except (ValueError, ZeroDivisionError):
        Ka = 10.0

    kaks_ratio = Ka / Ks if Ks > 0 else np.nan
    return Ka, Ks, kaks_ratio


# --- RSCU Codon Usage Bias ---
# Group codons by amino acid
_AA_TO_CODONS = {}
for _codon, _aa in standard_table.forward_table.items():
    _AA_TO_CODONS.setdefault(_aa, []).append(_codon)

def rscu_vector(codon_list):
    """Calculate a 59-dimensional RSCU vector (one per sense codon).
    RSCU = (observed freq of codon) / (expected freq if all synonymous codons
    used equally). Returns dict mapping codon -> RSCU value."""
    counts = Counter(c for c in codon_list if '-' not in c and len(c) == 3 and c in standard_table.forward_table)
    rscu = {}
    for aa, syn_codons in _AA_TO_CODONS.items():
        n_syn = len(syn_codons)
        total_aa = sum(counts.get(c, 0) for c in syn_codons)
        expected = total_aa / n_syn if n_syn > 0 else 0
        for c in syn_codons:
            rscu[c] = counts.get(c, 0) / max(1, expected)
    return rscu

def rscu_distance(codon_list1, codon_list2):
    """Euclidean distance between RSCU vectors of two sequences."""
    r1 = rscu_vector(codon_list1)
    r2 = rscu_vector(codon_list2)
    all_codons = sorted(set(r1.keys()) | set(r2.keys()))
    dist = math.sqrt(sum((r1.get(c, 0) - r2.get(c, 0))**2 for c in all_codons))
    return dist


# --- CAI (Codon Adaptation Index) ---
# Simple CAI weights (approximate relative adaptiveness for mammals)
_CAI_WEIGHTS = {
    'GCA': 0.58, 'GCC': 1.00, 'GCG': 0.26, 'GCT': 0.68, 'TGC': 1.00, 'TGT': 0.47,
    'GAC': 1.00, 'GAT': 0.43, 'GAA': 0.44, 'GAG': 1.00, 'TTC': 1.00, 'TTT': 0.43,
    'GGA': 0.44, 'GGC': 1.00, 'GGG': 0.43, 'GGT': 0.35, 'CAC': 1.00, 'CAT': 0.41,
    'ATA': 0.16, 'ATC': 1.00, 'ATT': 0.36, 'AAA': 0.43, 'AAG': 1.00, 'CTA': 0.18,
    'CTC': 1.00, 'CTG': 1.00, 'CTT': 0.33, 'TTA': 0.19, 'TTG': 0.33, 'ATG': 1.00,
    'AAC': 1.00, 'AAT': 0.44, 'CCA': 0.56, 'CCC': 1.00, 'CCG': 0.22, 'CCT': 0.53,
    'CAA': 0.26, 'CAG': 1.00, 'AGA': 0.21, 'AGG': 0.23, 'CGA': 0.11, 'CGC': 1.00,
    'CGG': 0.21, 'CGT': 0.09, 'AGC': 1.00, 'AGT': 0.32, 'TCA': 0.29, 'TCC': 0.74,
    'TCG': 0.14, 'TCT': 0.44, 'ACA': 0.55, 'ACC': 1.00, 'ACG': 0.17, 'ACT': 0.52,
    'GTA': 0.26, 'GTC': 1.00, 'GTG': 1.00, 'GTT': 0.28, 'TGG': 1.00, 'TAC': 1.00,
    'TAT': 0.40, 'TAA': 1.00, 'TAG': 1.00, 'TGA': 1.00
}

def calculate_cai(codon_list):
    weights = []
    for c in codon_list:
        if '-' not in c and len(c) == 3 and c in _CAI_WEIGHTS:
            weights.append(_CAI_WEIGHTS[c])
    if not weights:
        return 0.0
    # Geometric mean of weights
    # Use log to prevent underflow
    log_sum = sum(math.log(w) for w in weights if w > 0)
    return math.exp(log_sum / len(weights))

# --- Sliding Window Ka/Ks Variance ---
def sliding_kaks_variance(codon_list1, codon_list2, window_size=30, step=10):
    if len(codon_list1) < window_size:
        return np.nan
    
    kaks_values = []
    for i in range(0, len(codon_list1) - window_size + 1, step):
        w_c1 = codon_list1[i:i+window_size]
        w_c2 = codon_list2[i:i+window_size]
        _, _, kaks = nei_gojobori_kaks(w_c1, w_c2)
        kaks_values.append(kaks)
        
    if len(kaks_values) < 2:
        return np.nan
    # Filter out NaNs before calculating variance
    kaks_valid = [k for k in kaks_values if not math.isnan(k)]
    if len(kaks_valid) < 2:
        return np.nan
    return np.var(kaks_valid)

# --- Alignment Quality Scorer ---
def calculate_alignment_score(ref_seq, query_seq):
    matches = 0
    ref_gaps = 0
    query_gaps = 0

    for r, q in zip(ref_seq, query_seq):
        if r == '-': ref_gaps += 1
        elif q == '-': query_gaps += 1
        elif r == q: matches += 1

    length = len(ref_seq)
    total_gaps = ref_gaps + query_gaps
    aligned_positions = length - total_gaps

    identity_pct = (matches / aligned_positions * 100) if aligned_positions > 0 else 0
    gap_pct = (total_gaps / length * 100) if length > 0 else 0
    coverage_pct = (aligned_positions / length * 100) if length > 0 else 0

    gap_runs = []
    in_gap = False
    current_run = 0
    longest_ungapped = 0
    current_ungapped = 0

    for r, q in zip(ref_seq, query_seq):
        if r == '-' or q == '-':
            if in_gap: current_run += 1
            else: in_gap = True; current_run = 1
            current_ungapped = 0
        else:
            if in_gap:
                gap_runs.append(current_run)
                in_gap = False
                current_run = 0
            current_ungapped += 1
            longest_ungapped = max(longest_ungapped, current_ungapped)

    if in_gap: gap_runs.append(current_run)

    max_gap = max(gap_runs) if gap_runs else 0
    num_gap_runs = len(gap_runs)

    identity_score = (identity_pct / 100) * 60
    if coverage_pct < 10: identity_score *= 0.1
    elif coverage_pct < 20: identity_score *= 0.2
    elif coverage_pct < 30: identity_score *= 0.4
    elif coverage_pct < 50: identity_score *= 0.7

    if num_gap_runs < 10: gap_opening_penalty = 0
    elif num_gap_runs < 20: gap_opening_penalty = -3
    elif num_gap_runs < 30: gap_opening_penalty = -5
    else: gap_opening_penalty = -7

    if gap_pct < 10: gap_extension_penalty = 0
    elif gap_pct < 20: gap_extension_penalty = -2
    elif gap_pct < 30: gap_extension_penalty = -5
    elif gap_pct < 40: gap_extension_penalty = -8
    elif gap_pct < 50: gap_extension_penalty = -12
    elif gap_pct < 70: gap_extension_penalty = -18
    elif gap_pct < 90: gap_extension_penalty = -25
    else: gap_extension_penalty = -30

    if max_gap < 20: large_gap_penalty = 0
    elif max_gap < 50: large_gap_penalty = -((max_gap - 20) / 30) * 3
    elif max_gap < 100: large_gap_penalty = -3 - ((max_gap - 50) / 50) * 5
    elif max_gap < 150: large_gap_penalty = -8 - ((max_gap - 100) / 50) * 4
    else: large_gap_penalty = -12 - min(3, (max_gap - 150) / 50)

    if longest_ungapped >= 100: continuity_bonus = 25
    elif longest_ungapped >= 70: continuity_bonus = 20
    elif longest_ungapped >= 50: continuity_bonus = 17
    elif longest_ungapped >= 30: continuity_bonus = 15
    elif longest_ungapped >= 20: continuity_bonus = 12
    elif longest_ungapped >= 10: continuity_bonus = 8
    else: continuity_bonus = 5

    total_score = max(0, min(100, identity_score + gap_opening_penalty + gap_extension_penalty + large_gap_penalty + continuity_bonus))

    return {
        "align_score_total": total_score,
        "align_coverage": coverage_pct,
        "align_identity": identity_pct,
        "align_gap_pct": gap_pct,
        "align_max_gap": max_gap,
        "align_longest_ungapped": longest_ungapped
    }

POLARITY = {
    'G': 'nonpolar', 'A': 'nonpolar', 'V': 'nonpolar', 'L': 'nonpolar',
    'I': 'nonpolar', 'P': 'nonpolar', 'F': 'nonpolar', 'W': 'nonpolar',
    'M': 'nonpolar',
    'S': 'polar_uncharged', 'T': 'polar_uncharged', 'Y': 'polar_uncharged',
    'N': 'polar_uncharged', 'Q': 'polar_uncharged', 'C': 'polar_uncharged',
    'K': 'positive', 'R': 'positive', 'H': 'positive',
    'D': 'negative', 'E': 'negative'
}

def orf_dominance(seq):
    clean = seq.replace("-", "").upper()
    def longest_orf(s, frame):
        codons_f = [s[i:i+3] for i in range(frame, len(s)-2, 3)]
        max_run = run = 0
        for c in codons_f:
            if c in ("TAA", "TAG", "TGA"):
                run = 0
            else:
                run += 1
            max_run = max(max_run, run)
        return max_run
    f0 = longest_orf(clean, 0)
    f1 = longest_orf(clean, 1)
    f2 = longest_orf(clean, 2)
    total = max(1, f0 + f1 + f2)
    return f0 / total

def substitution_periodicity(seq1, seq2):
    signal = [1 if a == b else 0
              for a, b in zip(seq1, seq2)
              if a != '-' and b != '-']
    if len(signal) < 12:
        return {"period3_power": np.nan, "spectral_entropy": np.nan}
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))
    period3_idx = np.argmin(np.abs(freqs - 1/3))
    total_power = np.sum(fft_vals) + 1e-10
    period3_power = fft_vals[period3_idx] / total_power
    psd = fft_vals / total_power
    spectral_entropy = -np.sum(psd * np.log(psd + 1e-10))
    return {"period3_power": period3_power, "spectral_entropy": spectral_entropy}

def kmer_cosine(seq1, seq2, k):
    s1 = seq1.replace("-", "").upper()
    s2 = seq2.replace("-", "").upper()
    c1 = Counter(s1[i:i+k] for i in range(len(s1)-k+1))
    c2 = Counter(s2[i:i+k] for i in range(len(s2)-k+1))
    keys = set(c1) | set(c2)
    v1 = np.array([c1.get(kk, 0) for kk in keys], dtype=float)
    v2 = np.array([c2.get(kk, 0) for kk in keys], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return np.nan
    return float(np.dot(v1, v2) / denom)

def compute_enc(codon_list):
    counts = Counter(c for c in codon_list
                     if '-' not in c and len(c) == 3
                     and c in standard_table.forward_table)
    aa_codon_counts = {}
    for codon, count in counts.items():
        aa = AA_MAP.get(codon)
        if aa and aa != '*':
            if aa not in aa_codon_counts:
                aa_codon_counts[aa] = []
            aa_codon_counts[aa].append((codon, count))
    family_F = {}
    for aa, codon_counts in aa_codon_counts.items():
        n_synonymous = len(_AA_TO_CODONS.get(aa, [aa]))
        if n_synonymous <= 1:
            continue
        n_total = sum(c for _, c in codon_counts)
        if n_total < 2:
            continue
        pi_sq = sum((c / n_total) ** 2 for _, c in codon_counts)
        F = (n_total * pi_sq - 1) / (n_total - 1)
        if n_synonymous not in family_F:
            family_F[n_synonymous] = []
        family_F[n_synonymous].append(F)
    if not family_F:
        return np.nan
    enc = 0.0
    for ns, Fs in family_F.items():
        mean_F = np.mean(Fs)
        if mean_F > 0:
            enc += ns / mean_F
    return min(enc, 61.0)

def gc_content(seq):
    clean = seq.replace("-", "").upper()
    if not clean:
        return np.nan
    return (clean.count("G") + clean.count("C")) / len(clean)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(seq1, seq2, transcript_id=None, esm_cache=None, exon_number=None, total_exons=None):
    c1 = codons(seq1.upper())
    c2 = codons(seq2.upper())

    # Safely align lengths if they don't match exactly by padding or truncating
    min_len = min(len(c1), len(c2))
    c1 = c1[:min_len]
    c2 = c2[:min_len]
    L = min_len

    # Deep learning features lookup
    esm_feats = {
        "esm_cosine": 0,
        "esm_l2": 0,
        "esm_length_ratio": 0,
        "esm_log_likelihood_diff": 0,
        "esm_human_log_likelihood": 0,
        "esm_mouse_log_likelihood": 0
    }
    
    if esm_cache and transcript_id in esm_cache:
        t_data = esm_cache[transcript_id]
        h_data = t_data.get('human', {})
        m_data = t_data.get('mouse', {})
        anchor = t_data.get('global_human_mean', np.zeros(1280))
        
        h_vec = h_data.get('mean')
        m_vec = m_data.get('mean')
        
        if h_vec is not None and m_vec is not None:
            # Mean center to fix anisotropy
            h_vec_cent = h_vec - anchor
            m_vec_cent = m_vec - anchor
            
            # Cosine similarity
            norm_h = np.linalg.norm(h_vec_cent)
            norm_m = np.linalg.norm(m_vec_cent)
            if norm_h > 0 and norm_m > 0:
                esm_feats["esm_cosine"] = np.dot(h_vec_cent, m_vec_cent) / (norm_h * norm_m)
            
            # L2 Distance
            esm_feats["esm_l2"] = np.linalg.norm(h_vec_cent - m_vec_cent)
            
            # Length ratio (captures gross insertion/deletion/early-stop)
            h_len = h_data.get('len', 0)
            m_len = m_data.get('len', 0)
            esm_feats["esm_length_ratio"] = m_len / max(1, h_len)
            
            # Log Likelihoods (predicts mutational constraint)
            h_ll = h_data.get('log_likelihood', 0)
            m_ll = m_data.get('log_likelihood', 0)
            esm_feats["esm_human_log_likelihood"] = h_ll
            esm_feats["esm_mouse_log_likelihood"] = m_ll
            esm_feats["esm_log_likelihood_diff"] = h_ll - m_ll

    if L == 0:
        placeholder = {
            "ka": 0, "ks": 0, "kaks_ratio": 0,
            "nonsyn_rate": 0, "transition_ratio": 0, "radical_frac": 0,
            "hydro_mean": 0, "charge_mean": 0, "premature_stops": 0, "frameshift_frac": 0,
            "frame_preserving_count": 0, "pct_intact_orf": 0, "padding_ratio": 0,
            "four_fold_id": 0,
            "gc3_diff": 0, "grantham_mean": 0, "grantham_max": 0, "nonsyn_spatial_entropy": 0,
            "terminal_gap_bias": 0, "cpg_tpg_rate": 0,
            "gap_open": 0, "gap_mean": 0, "run_mean": 0, "run_max": 0,
            "rscu_distance": 0,
            "align_gap_pct": 0, "align_longest_ungapped": 0,
            "orf_dominance_human": 0, "orf_dominance_mouse": 0, "orf_dominance_diff": 0,
            "period3_power": 0, "spectral_entropy": 0,
            "kmer4_cosine": 0,
            "enc_diff": 0,
            "polarity_preserving_frac": 0, "gc_diff_overall": 0,
            "exon_position_relative": 0, "is_first_exon": 0,
        }
        placeholder.update(esm_feats)
        return np.array(list(placeholder.values())), list(placeholder.keys())

    syn = nonsyn = 0
    transitions = transversions = 0
    radical = 0
    
    hydro_diffs = []
    charge_diffs = []
    grantham_scores = []
    
    gaps_human = []
    gaps_mouse = []
    current_gap_human = 0
    current_gap_mouse = 0
    
    nonsyn_runs = []
    nonsyn_aa_pairs = []
    nonsyn_positions = []
    run = 0

    aa_ident = 0
    
    # Codon position identities
    nt_ident_pos = [0, 0, 0]
    # 4-fold degenerate tracker
    four_fold_total = 0
    four_fold_matches = 0

    # CpG transitions
    cpg_total = 0
    cpg_tpg_transitions = 0
    
    # GC3 counters
    gc3_human = 0
    gc3_mouse = 0
    valid_gc3_codons_human = 0
    valid_gc3_codons_mouse = 0

    # Premature stop counter
    premature_stops = 0
    
    for i, (cod1, cod2) in enumerate(zip(c1, c2)):
        
        # Identity per position
        for pos in range(min(len(cod1), len(cod2))):
            if cod1[pos] != '-' and cod1[pos] == cod2[pos]:
                nt_ident_pos[pos] += 1
                
        # 4-fold degenerate check
        if cod1 in FOUR_FOLD_CODONS:
            if '-' not in cod1 and '-' not in cod2:
                four_fold_total += 1
                if cod1[2] == cod2[2]:
                    four_fold_matches += 1
                    
        # GC3 check
        if len(cod1) == 3 and cod1[2] in 'GC': gc3_human += 1
        if len(cod1) == 3 and '-' not in cod1: valid_gc3_codons_human += 1
        
        if len(cod2) == 3 and cod2[2] in 'GC': gc3_mouse += 1
        if len(cod2) == 3 and '-' not in cod2: valid_gc3_codons_mouse += 1
        
        # Premature stops
        if cod2 in ['TAA', 'TAG', 'TGA']:
            premature_stops += 1
            
        # Gaps tracking
        if "-" in cod1:
            current_gap_human += 1
        else:
            if current_gap_human:
                gaps_human.append((i, current_gap_human))
                current_gap_human = 0
                
        if "-" in cod2:
            current_gap_mouse += 1
            if run:
                nonsyn_runs.append(run)
                run = 0
        else:
            if current_gap_mouse:
                gaps_mouse.append((i, current_gap_mouse))
                current_gap_mouse = 0

        res = classify_substitution(cod1, cod2)
        if res is None:
            continue

        s, ti, tv, rad, aa1, aa2 = res
        transitions += ti
        transversions += tv

        if aa1 == aa2:
            aa_ident += 1

        if s:
            syn += 1
            if run:
                nonsyn_runs.append(run)
                run = 0
        else:
            nonsyn += 1
            nonsyn_positions.append(i)
            nonsyn_aa_pairs.append((aa1, aa2))
            run += 1

        if not s:
            hydro_diffs.append(abs(hydrophobicity.get(aa1,0) - hydrophobicity.get(aa2,0)))
            charge_diffs.append(abs(aa_charge(aa1) - aa_charge(aa2)))
            grantham_scores.append(get_grantham(aa1, aa2))
            if rad:
                radical += 1

    if run: nonsyn_runs.append(run)
    if current_gap_human: gaps_human.append((L, current_gap_human))
    if current_gap_mouse: gaps_mouse.append((L, current_gap_mouse))

    # Gap metrics
    gaps_combined_lengths = [g[1] for g in gaps_mouse] + [g[1] for g in gaps_human]
    total_gaps = len(gaps_combined_lengths)
    
    frameshifts = sum(1 for g in gaps_combined_lengths if g % 3 != 0)
    frame_preserving = sum(1 for g in gaps_combined_lengths if g % 3 == 0)
    
    # Terminal gap bias (first or last 10% of alignment)
    terminal_gaps = sum(1 for g in gaps_mouse + gaps_human if g[0] <= 0.1 * L or g[0] >= 0.9 * L)
    
    # Intact reading frame
    # Longest stretch without frameshifts or premature stops
    intact_stretches = []
    current_stretch = 0
    for cod in c2:
        if cod in ['TAA', 'TAG', 'TGA'] or ("-" in cod and cod.count("-") % 3 != 0):
            intact_stretches.append(current_stretch)
            current_stretch = 0
        else:
            current_stretch += 1
    intact_stretches.append(current_stretch)
    pct_intact_orf = max(intact_stretches) / max(1, L)
    
    # CpG -> TpG transitions (using the raw sequences)
    seq1_clean = seq1.upper()
    seq2_clean = seq2.upper()
    for i in range(len(seq1_clean) - 1):
        if seq1_clean[i:i+2] == 'CG':
            cpg_total += 1
            if seq2_clean[i:i+2] in ['TG', 'CA']:
                cpg_tpg_transitions += 1

    length = max(1, L)
    ungapped_len1 = len(seq1.replace('-', ''))
    ungapped_len2 = len(seq2.replace('-', ''))
    
    # Nei-Gojobori Ka/Ks (replaces crude omega)
    ka, ks, kaks = nei_gojobori_kaks(c1, c2)
    
    # RSCU codon usage bias distance
    rscu_dist = rscu_distance(c1, c2)

    features = {
        # Evolutionary Rates (Nei-Gojobori with Jukes-Cantor correction)
        "ka": ka,
        "ks": ks,
        "kaks_ratio": kaks,
        "kaks_window_variance": sliding_kaks_variance(c1, c2),
        "nonsyn_rate": nonsyn/length if length > 0 else np.nan,
        "transition_ratio": transitions/max(1, transitions+transversions) if (transitions+transversions) > 0 else np.nan,
        "radical_frac": radical/max(1, nonsyn) if nonsyn > 0 else np.nan,
        "hydro_mean": np.mean(hydro_diffs) if hydro_diffs else 0,
        "charge_mean": np.mean(charge_diffs) if charge_diffs else 0,
        
        # High Impact Lesion Features
        "premature_stops": premature_stops,
        "frameshift_frac": frameshifts / max(1, total_gaps),
        "frame_preserving_count": frame_preserving,
        "pct_intact_orf": pct_intact_orf,
        "padding_ratio": ungapped_len2 / max(1, ungapped_len1),
        
        # Position / Bias Features
        "four_fold_id": four_fold_matches / max(1, four_fold_total),
        "gc3_diff": abs((gc3_human/max(1, valid_gc3_codons_human)) - (gc3_mouse/max(1, valid_gc3_codons_mouse))),
        "cai_ratio": calculate_cai(c2) / max(0.001, calculate_cai(c1)),
        
        # Grantham Scores
        "grantham_mean": np.mean(grantham_scores) if grantham_scores else 0,
        
        # Entropy & Global Sequence Stats
        "nonsyn_spatial_entropy": shannon_entropy(nonsyn_positions, length),
        "terminal_gap_bias": terminal_gaps / max(1, total_gaps),
        "cpg_tpg_rate": cpg_tpg_transitions / max(1, cpg_total),
        
        "gap_open": len(gaps_mouse)/length,
        "gap_mean": np.mean(gaps_combined_lengths) if gaps_combined_lengths else 0,
        "run_mean": np.mean(nonsyn_runs) if nonsyn_runs else 0,
        "run_max": max(nonsyn_runs) if nonsyn_runs else 0,
        
        # New Phase 4 features
        "rscu_distance": rscu_dist,
    }
    
    # Add new alignment quality features (drop coverage as it's -1.0 w/ gap_pct)
    align_metrics = calculate_alignment_score(seq1.upper()[:min_len*3], seq2.upper()[:min_len*3])
    features["align_gap_pct"] = align_metrics["align_gap_pct"]
    features["align_longest_ungapped"] = align_metrics["align_longest_ungapped"]

    # --- Group 1 Features ---

    # Feature 1: ORF Dominance
    od_h = orf_dominance(seq1)
    od_m = orf_dominance(seq2)
    features["orf_dominance_human"] = od_h
    features["orf_dominance_mouse"] = od_m
    features["orf_dominance_diff"] = abs(od_h - od_m)

    # Feature 2: Spectral Periodicity
    spec = substitution_periodicity(seq1, seq2)
    features["period3_power"] = spec["period3_power"]
    features["spectral_entropy"] = spec["spectral_entropy"]

    # Feature 3: k-mer Cosine Similarity
    features["kmer4_cosine"] = kmer_cosine(seq1, seq2, 4)

    # Feature 4: ENC Difference
    enc_h = compute_enc(c1)
    enc_m = compute_enc(c2)
    features["enc_diff"] = np.nan if (np.isnan(enc_h) or np.isnan(enc_m)) else abs(enc_h - enc_m)

    # Feature 6: Polarity-Preserving Substitution Fraction
    if nonsyn_aa_pairs:
        polarity_preserved = sum(
            1 for a1, a2 in nonsyn_aa_pairs
            if POLARITY.get(a1) == POLARITY.get(a2)
            and POLARITY.get(a1) is not None
        )
        features["polarity_preserving_frac"] = polarity_preserved / len(nonsyn_aa_pairs)
    else:
        features["polarity_preserving_frac"] = np.nan

    # Feature 7: GC Content Overall Difference
    features["gc_diff_overall"] = abs(gc_content(seq1) - gc_content(seq2))

    # Exon position features
    features["exon_position_relative"] = (exon_number / total_exons) if (exon_number is not None and total_exons and total_exons > 0) else 0.5
    features["is_first_exon"] = int(exon_number == 1) if exon_number is not None else 0

    features.update(esm_feats)
    return np.array(list(features.values())), list(features.keys())
