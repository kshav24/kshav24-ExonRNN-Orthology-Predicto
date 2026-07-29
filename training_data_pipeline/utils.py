#!/usr/bin/env python3
"""
utils.py — CESAR pipeline utilities
"""

import subprocess
import os
import tempfile


# ── Subprocess ────────────────────────────────────────────────────────────────

def run_subprocess(command, return_output=False):
    """
    Run a shell command. Uses shell=True so paths with colons/spaces work
    and so mafExtract/twoBitToFa flags are parsed correctly.
    """
    env = os.environ.copy()
    env["PATH"] = "/home/keshav/ucsc_tools:/home/keshav/CESAR2.0/tools:" + env.get("PATH", "")

    result = subprocess.run(
        command, shell=True, capture_output=True, env=env)

    stdout_text = result.stdout.decode('utf-8', errors='replace')
    stderr_text = result.stderr.decode('utf-8', errors='replace')

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{stderr_text}")

    if return_output:
        return stdout_text.strip()


# ── Temp files ────────────────────────────────────────────────────────────────

def create_temp_file():
    """Create a temp file in /dev/shm (RAM-backed, fast)."""
    fd, path = tempfile.mkstemp(dir='/dev/shm', prefix='cesar_')
    os.close(fd)
    return path


# ── FASTA / sequence ──────────────────────────────────────────────────────────

def fasta_to_dict(fasta_file):
    """Parse FASTA → {header: sequence}. Header is text after '>' up to first space."""
    if not os.path.isfile(fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    result = {}
    seq_id, seq = None, []
    with open(fasta_file) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if seq_id is not None:
                    result[seq_id] = ''.join(seq)
                seq_id = line[1:].split()[0]   # first word after '>'
                seq = []
            else:
                seq.append(line)
    if seq_id is not None:
        result[seq_id] = ''.join(seq)
    return result


def reverse_comp(sequence):
    """Reverse complement of a DNA sequence (handles upper and lower case)."""
    table = str.maketrans('ATGCatgcNn', 'TACGtacgNn')
    return sequence.translate(table)[::-1]


# ── 2bit sequence extraction ──────────────────────────────────────────────────

def get_sequence_from_2bit(start, end, chrom, two_bit_file):
    """
    Extract sequence from a 2bit file.
    Coordinates are 0-based half-open [start, end), same as twoBitToFa convention.
    """
    if start < 0:
        start = 0
    if start >= end:
        raise ValueError(f"Invalid coordinates: {chrom}:{start}-{end}")
    cmd = f"twoBitToFa {two_bit_file}:{chrom}:{start}-{end} stdout"
    raw = run_subprocess(cmd, return_output=True)
    return ''.join(line for line in raw.split('\n') if not line.startswith('>'))


# ── MAF extraction & filtering ────────────────────────────────────────────────

def run_maf_extract(maf_index, output_file, chromosome, exon):
    """
    Run mafExtract for one exon region.
    exon: "start-end" string (0-based half-open coordinates).
    """
    start, end = exon.split('-')
    cmd = f"mafExtract {maf_index} -region={chromosome}:{start}-{end} {output_file}"
    run_subprocess(cmd)


def filter_maf_file(input_maf, output_maf, species_list):
    """
    Filter a MAF file to keep only alignment blocks that contain at least one
    species from species_list.  Within each kept block, all s-lines are written
    (ref + query species); i-lines and e-lines are discarded.
    """
    species_set = set(species_list)

    with open(input_maf) as fi, open(output_maf, 'w') as fo:
        block_lines     = []
        block_has_query = False

        for line in fi:
            stripped = line.strip()

            # Pass through file-level header comments
            if stripped.startswith('##') or (stripped.startswith('#') and not block_lines):
                fo.write(line)
                continue

            # Blank line = end of alignment block
            if stripped == '':
                if block_has_query:
                    fo.writelines(block_lines)
                    fo.write('\n')
                block_lines = []
                block_has_query = False
                continue

            # Start of a new alignment block
            if stripped.startswith('a'):
                block_lines     = [line]
                block_has_query = False

            # Sequence lines only (skip i-lines, e-lines, q-lines)
            elif stripped.startswith('s'):
                parts = stripped.split()
                if len(parts) >= 7 and '.' in parts[1]:
                    sp = parts[1].split('.')[0]
                    if sp in species_set:
                        block_has_query = True
                block_lines.append(line)

        # Flush last block
        if block_lines and block_has_query:
            fo.writelines(block_lines)

def get_sline_data(line):
    """
    Parse an 's' line from a MAF file.
    Format: s src start size strand srcSize text
    """
    parts = line.split()
    src = parts[1]
    if '.' in src:
        species, chromosome = src.split('.', 1)
    else:
        species, chromosome = src, src
    start = int(parts[2])
    size = int(parts[3])
    strand = parts[4]
    src_size = int(parts[5])
    seq = parts[6]
    return species, chromosome, start, size, strand, src_size, seq

# Alias for backwards compatibility with geneloss logic
create_temp_devshm = create_temp_file


# ── GenePred parsing ──────────────────────────────────────────────────────────

def parse_genepred_line(line):
    """
    Parse one GenePred-format line.
    Expected columns: transcript_id, chrom, strand, gene_start, gene_end,
                      cds_start, cds_end, exons (comma-separated start-end pairs)
    """
    fields = line.strip().split('\t')
    if len(fields) < 8:
        raise ValueError(f"Invalid GenePred line: {line.strip()}")
    exon_coords = []
    for pair in fields[7].split(','):
        pair = pair.strip()
        if pair and '-' in pair:
            s, e = map(int, pair.split('-'))
            exon_coords.append(f"{s}-{e}")
    return {
        'transcript_id': fields[0],
        'chromosome':    fields[1],
        'strand':        fields[2],
        'gene_start':    int(fields[3]),
        'gene_end':      int(fields[4]),
        'cds_start':     int(fields[5]),
        'cds_end':       int(fields[6]),
        'exons':         exon_coords,
    }


def load_genepred_file(genepred_file, transcript_ids=None):
    """Load GenePred file → {transcript_id: gene_info dict}."""
    transcripts = {}
    with open(genepred_file) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            try:
                info = parse_genepred_line(line)
                tid  = info['transcript_id']
                if transcript_ids and tid not in transcript_ids:
                    continue
                transcripts[tid] = info
            except ValueError:
                continue
    return transcripts
