#!/usr/bin/env python3
"""
generate_training_data.py
─────────────────────────
Runs CESAR for ortholog/paralog pairs and saves RAW alignments only.
No feature computation — that's feature_extraction_pipeline.py's job.

Output CSV columns:
  transcript_id, exon_number, species, ref_seq, query_seq, label

Usage:
  python generate_training_data.py \
    one_to_one_orthologs.csv \
    ortholog_paralog_pair_negatives.txt \
    ensg_to_enst.tsv \
    updatedensGene.gp \
    hg38.5way.bb hg38 2bit_dir/ /CESAR2.0/ \
    raw_train.csv raw_test.csv \
    --species mm10 bosTau8 equCab2 canFam3
"""

import os
import re
import csv
import random
import argparse
import multiprocessing
import functools

from cesar_runner import CesarRunner
import utils

# ─────────────────────────────────────────────────────────────────────────────
# Alignment splitter
# ─────────────────────────────────────────────────────────────────────────────

def _strip_phase(r, q):
    p5 = 0
    while p5 < len(r) and r[p5].islower():
        p5 += 1
    p3 = len(r)
    while p3 > p5 and r[p3 - 1].islower():
        p3 -= 1
    return r[p5:p3].upper(), q[p5:p3].upper()


def split_and_clean_multiexon(ref_seq, query_seq):
    """Split CESAR multi-exon output into per-exon (ref, query) pairs."""
    DELIM   = '>' * 19
    results = []

    def _mask_strip(r_part, q_part):
        rc, qc = [], []
        for r, q in zip(r_part, q_part):
            if r == ' ':
                continue
            rc.append(r); qc.append(q)
        return _strip_phase(''.join(rc), ''.join(qc))

    if DELIM in ref_seq:
        ref_seq   = ref_seq.replace(DELIM, ' ')
        query_seq = query_seq.replace(DELIM, ' ')

    parts = ref_seq.split(' ')
    idx   = 0
    for r_part in parts:
        q_part     = query_seq[idx : idx + len(r_part)]
        r_ex, q_ex = _mask_strip(r_part, q_part)
        idx       += len(r_part) + 1
        if not r_ex.strip('-'):
            continue
        results.append((r_ex, q_ex))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Single-species CESAR runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_species(transcript, gene_info, reference, query_sp,
                     twobit_dir, cesar_dir, alignment, clade,
                     neg_locus_info=None, debug_dir=None):
    runner = CesarRunner(
        transcript, gene_info, reference, [query_sp],
        twobit_dir, cesar_dir, clade, False
    )
    runner.neg_locus_info = neg_locus_info
    runner.debug_dir      = debug_dir

    result = runner.run_multi_exon(alignment)
    if not result:
        return None

    return split_and_clean_multiexon(result['ref_seq'], result['query_seq'])

# ─────────────────────────────────────────────────────────────────────────────
# Public CESAR runners
# ─────────────────────────────────────────────────────────────────────────────

def run_transcript(transcript, genepred, alignment, reference,
                   species_list, twobit_dir, cesar_dir, clade='human'):
    """Positive mode — returns {exon_idx: {species: (ref, query)}}"""
    try:
        gene_data = utils.load_genepred_file(genepred, {transcript})
    except Exception as e:
        print(f'  [ERR] load_genepred {transcript}: {e}')
        return None

    if transcript not in gene_data:
        return None

    gene_info = gene_data[transcript]
    per_exon  = {}
    for sp in species_list:
        pairs = _run_one_species(transcript, gene_info, reference, sp,
                                 twobit_dir, cesar_dir, alignment, clade)
        if pairs is None:
            continue
        for i, pair in enumerate(pairs, 1):
            per_exon.setdefault(i, {})[sp] = pair
    return per_exon or None


def run_negative_pair(paralog_transcript, ortholog_transcript,
                      genepred, alignment, reference,
                      species_list, twobit_dir, cesar_dir,
                      clade='human', debug_dir=None):
    """Negative mode — returns {exon_idx: {species: (ref, query)}}"""
    try:
        gene_data = utils.load_genepred_file(
            genepred, {paralog_transcript, ortholog_transcript}
        )
    except Exception as e:
        print(f'  [ERR] load_genepred {paralog_transcript}: {e}')
        return None

    if paralog_transcript not in gene_data or ortholog_transcript not in gene_data:
        return None

    para_info = gene_data[paralog_transcript]
    orth_info = gene_data[ortholog_transcript]
    per_exon  = {}
    for sp in species_list:
        pairs = _run_one_species(paralog_transcript, para_info, reference, sp,
                                 twobit_dir, cesar_dir, alignment, clade,
                                 neg_locus_info=orth_info, debug_dir=debug_dir)
        if pairs is None:
            continue
        for i, pair in enumerate(pairs, 1):
            per_exon.setdefault(i, {})[sp] = pair
    return per_exon or None

# ─────────────────────────────────────────────────────────────────────────────
# Raw CSV writer
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ['transcript_id', 'exon_number', 'species',
          'ref_seq', 'query_seq', 'label']

def write_raw_csv(target_csv, transcript, label, per_exon, species_list):
    """Write one row per (exon x species)."""
    file_exists = os.path.isfile(target_csv)
    with open(target_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        for exon_idx in sorted(per_exon):
            sp_pairs = per_exon[exon_idx]
            for sp in species_list:
                if sp in sp_pairs:
                    ref, query = sp_pairs[sp]
                else:
                    ref, query = '', ''
                writer.writerow({
                    'transcript_id': transcript,
                    'exon_number'  : exon_idx,
                    'species'      : sp,
                    'ref_seq'      : ref,
                    'query_seq'    : query,
                    'label'        : label,
                })

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_ensg_enst_map(ensg_enst_file):
    """Read ensg_to_enst.tsv. Returns {ENSG_id: ENST_id}"""
    mapping = {}
    with open(ensg_enst_file, 'r') as f:
        for line in f:
            if line.startswith('ensg') or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0] and parts[1]:
                mapping[parts[0]] = parts[1]
    print(f'  [map] Loaded {len(mapping)} ENSG→ENST mappings')
    return mapping


def load_positives(orthologs_csv, ensg_enst_map):
    """Read one_to_one_orthologs.csv. Returns list of ENST IDs."""
    positives = []
    skipped   = 0
    with open(orthologs_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ensg = row.get('human_gene_id', '').strip()
            if not ensg:
                continue
            enst = ensg_enst_map.get(ensg)
            if enst:
                positives.append(enst)
            else:
                skipped += 1
    print(f'  [pos] {len(positives)} positive transcripts resolved, {skipped} skipped (no MANE transcript)')
    return positives


def load_neg_pairs(neg_pairs_file, ensg_enst_map):
    """Parse ortholog_paralog_pair_negatives.txt. Returns list of (ENST_paralog, ENST_ortholog)."""
    resolved, skipped = [], 0
    with open(neg_pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('-')
            if len(parts) != 2:
                skipped += 1
                continue
            ensg_orth, ensg_para = parts[0], parts[1]
            enst_orth = ensg_enst_map.get(ensg_orth)
            enst_para = ensg_enst_map.get(ensg_para)
            if enst_orth and enst_para:
                resolved.append((enst_para, enst_orth))
            else:
                skipped += 1
    print(f'  [neg] {len(resolved)} pairs resolved, {skipped} skipped')
    return resolved

# ─────────────────────────────────────────────────────────────────────────────
# Workers
# ─────────────────────────────────────────────────────────────────────────────

def worker_positive(item, args):
    enst, label = item
    try:
        per_exon = run_transcript(
            transcript   = enst,
            genepred     = args.genepred,
            alignment    = args.alignment,
            reference    = args.reference,
            species_list = args.species,
            twobit_dir   = args.twobit_dir,
            cesar_dir    = args.cesar_dir,
        )
        return enst, label, per_exon
    except Exception as e:
        print(f'  [ERR pos] {enst}: {e}')
        return enst, label, None


def worker_negative(item, args):
    enst_para, enst_orth = item
    try:
        per_exon = run_negative_pair(
            paralog_transcript  = enst_para,
            ortholog_transcript = enst_orth,
            genepred     = args.genepred,
            alignment    = args.alignment,
            reference    = args.reference,
            species_list = args.species,
            twobit_dir   = args.twobit_dir,
            cesar_dir    = args.cesar_dir,
            debug_dir    = args.debug_dir,
        )
        return enst_para, 0, per_exon
    except Exception as e:
        print(f'  [ERR neg] para={enst_para}: {e}')
        return enst_para, 0, None

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate raw CESAR alignments for training data'
    )
    parser.add_argument('orthologs_csv',   help='one_to_one_orthologs.csv (positive gene source)')
    parser.add_argument('neg_pairs_file',  help='ortholog_paralog_pair_negatives.txt')
    parser.add_argument('ensg_enst_file',  help='ensg_to_enst.tsv (MANE ENSG→ENST map)')
    parser.add_argument('genepred',        help='updatedensGene.gp')
    parser.add_argument('alignment',       help='hg38.5way.bb')
    parser.add_argument('reference',       help='Reference species, e.g. hg38')
    parser.add_argument('twobit_dir',      help='Directory with .2bit files')
    parser.add_argument('cesar_dir',       help='CESAR2.0 directory')
    parser.add_argument('output_train',    help='Output raw train CSV')
    parser.add_argument('output_test',     help='Output raw test CSV')

    parser.add_argument('--species', nargs='+',
                        default=['mm10', 'bosTau8', 'equCab2', 'canFam3'])
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--positives',  type=int,   default=2000)
    parser.add_argument('--negatives',  type=int,   default=2000)
    parser.add_argument('--all',        action='store_true')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--debug_dir',  default=None)
    parser.add_argument('--cores', type=int,
                        default=min(15, multiprocessing.cpu_count()))

    args = parser.parse_args()
    random.seed(args.seed)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    print('\n=== Loading input files ===')
    ensg_enst_map   = load_ensg_enst_map(args.ensg_enst_file)
    pos_transcripts = load_positives(args.orthologs_csv, ensg_enst_map)
    neg_pairs       = load_neg_pairs(args.neg_pairs_file, ensg_enst_map)

    if args.all:
        sampled_pos = pos_transcripts
        sampled_neg = neg_pairs
    else:
        sampled_pos = random.sample(pos_transcripts, min(args.positives, len(pos_transcripts)))
        sampled_neg = random.sample(neg_pairs,       min(args.negatives, len(neg_pairs)))

    random.shuffle(sampled_pos)
    random.shuffle(sampled_neg)

    pos_split = int(len(sampled_pos) * (1 - args.test_split))
    neg_split = int(len(sampled_neg) * (1 - args.test_split))

    train_pos = sampled_pos[:pos_split];  test_pos = sampled_pos[pos_split:]
    train_neg = sampled_neg[:neg_split];  test_neg = sampled_neg[neg_split:]

    print(f'\n=== Dataset split (seed={args.seed}) ===')
    print(f'  Train: {len(train_pos)} pos + {len(train_neg)} neg')
    print(f'  Test:  {len(test_pos)} pos + {len(test_neg)} neg')

    split_map = {}
    for t in train_pos:         split_map[t]   = args.output_train
    for t in test_pos:          split_map[t]   = args.output_test
    for (par, _) in train_neg:  split_map[par] = args.output_train
    for (par, _) in test_neg:   split_map[par] = args.output_test

    for f in [args.output_train, args.output_test]:
        if os.path.exists(f):
            os.remove(f)

    # Positives
    pos_items   = [(t, 1) for t in (train_pos + test_pos)]
    pos_fn      = functools.partial(worker_positive, args=args)
    pos_success = 0

    print(f'\n=== Processing {len(pos_items)} positives ({args.cores} cores) ===')
    with multiprocessing.Pool(processes=args.cores) as pool:
        for idx, (transcript, label, per_exon) in enumerate(
                pool.imap_unordered(pos_fn, pos_items), 1):
            if idx % 200 == 0 or idx == len(pos_items):
                print(f'  [{idx:>5}/{len(pos_items)}] success={pos_success}')
            if per_exon:
                pos_success += 1
                write_raw_csv(split_map[transcript], transcript, label,
                              per_exon, args.species)

    # Negatives
    neg_items   = list(train_neg) + list(test_neg)
    neg_fn      = functools.partial(worker_negative, args=args)
    neg_success = 0

    print(f'\n=== Processing {len(neg_items)} negatives ({args.cores} cores) ===')
    with multiprocessing.Pool(processes=args.cores) as pool:
        for idx, (transcript, label, per_exon) in enumerate(
                pool.imap_unordered(neg_fn, neg_items), 1):
            if idx % 200 == 0 or idx == len(neg_items):
                print(f'  [{idx:>5}/{len(neg_items)}] success={neg_success}')
            if per_exon:
                neg_success += 1
                write_raw_csv(split_map[transcript], transcript, label,
                              per_exon, args.species)

    print(f'\n=== Done ===')
    print(f'  Positives: {pos_success}/{len(pos_items)} succeeded')
    print(f'  Negatives: {neg_success}/{len(neg_items)} succeeded')
    print(f'  Train CSV → {args.output_train}')
    print(f'  Test  CSV → {args.output_test}')


if __name__ == '__main__':
    main()
