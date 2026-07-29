#!/usr/bin/env python3
"""
CESAR Runner - always runs in multi-exon mode.

Two optional attributes (set AFTER construction, not as kwargs):
    runner.neg_locus_info  = ortholog gene_info   -> negative pair mode
    runner.debug_dir       = '/tmp/cesar_debug'   -> save per-species debug files
"""

import os
import shutil
from refExon            import ReferenceExon
from queryExon          import QueryExon
import utils
# from cesarPostProcessor import CesarOutputPostProcessor


class CesarRunner:

    def __init__(self, transcript_id, gene_info, ref_species,
                 query_species_list, two_bit_dir, cesar_dir,
                 clade, verbose=False):
        self.transcript_id      = transcript_id
        self.gene_info          = gene_info
        self.ref_species        = ref_species
        self.query_species_list = query_species_list
        self.two_bit_dir        = two_bit_dir
        self.cesar_dir          = cesar_dir
        self.clade              = clade
        self.verbose            = verbose
        self.run_post_process   = True

        # set after construction if needed
        self.neg_locus_info = None
        self.debug_dir      = None

    # ── public ────────────────────────────────────────────────────────────────

    def run_multi_exon(self, alignment_file):
        return self._run_cesar(alignment_file)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _debug_prefix(self):
        sp_tag = '_'.join(self.query_species_list)
        if self.neg_locus_info is not None:
            orth_id = self.neg_locus_info.get('transcript_id', None)
            if orth_id is None:
                q_chrom = self.neg_locus_info.get('chromosome', 'unk')
                q_start = self.neg_locus_info['exons'][0][0]
                orth_id = f'{q_chrom}_{q_start}'
            return f'NEG_{self.transcript_id}_vs_{orth_id}_{sp_tag}'
        return f'POS_{self.transcript_id}_{sp_tag}'

    def _save_debug(self, src, suffix):
        dest_dir = self.debug_dir or '/tmp'
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f'{self._debug_prefix()}{suffix}')
        shutil.copyfile(src, dest)
        return dest

    # ── core ──────────────────────────────────────────────────────────────────

    def _run_cesar(self, alignment_file):

        # ── reference side ────────────────────────────────────────────────────
        chromosome = self.gene_info['chromosome']
        strand     = self.gene_info['strand']
        exons_list = list(self.gene_info['exons'])
        num_exons  = len(exons_list)
        if strand == '-':
            exons_list = exons_list[::-1]   # biological order for ref (CESAR needs 5'->3')

        # ── query locus ───────────────────────────────────────────────────────
        # QueryExon sorts exons by genomic coord internally — pass raw order, NO flip
        locus        = self.neg_locus_info if self.neg_locus_info is not None else self.gene_info
        locus_chrom  = locus['chromosome']
        locus_strand = locus['strand']
        locus_exons  = list(locus['exons'])   # keep genomic order — QueryExon handles it

        if self.verbose:
            print(f'  Running MULTI-EXON mode...')
            if self.neg_locus_info is not None:
                print(f'  [NEGATIVE MODE] ref exons from {chromosome} | '
                      f'query locus from {locus_chrom}')

        # ── reference sequences ───────────────────────────────────────────────
        ref_two_bit  = os.path.join(self.two_bit_dir, self.ref_species,
                                    f'{self.ref_species}.2bit')
        ref_exon_obj = ReferenceExon(exons_list, chromosome, strand, ref_two_bit)
        ref_seqs, splice_sites = ref_exon_obj.extract_for_cesar()

        # ── query sequences ───────────────────────────────────────────────────
        query_exon_obj = QueryExon(
            self.ref_species, locus_exons, locus_chrom, locus_strand,
            self.two_bit_dir, self.query_species_list,
            alignment_file, self.cesar_dir
        )
        try:
            query_seqs = query_exon_obj.extract_query_sequences(False)
        except Exception as e:
            if self.verbose:
                print(f'  QueryExon FAILED: {e}')
            return None

        if not query_seqs:
            if self.verbose:
                print('  QueryExon returned empty — no colinear mapping found.')
            return None

        # ── build CESAR input / output temp files ─────────────────────────────
        cesar_input  = utils.create_temp_file()
        cesar_output = utils.create_temp_file()
        result       = None

        try:
            # THIS PASSES locus_strand TO THE WRITER
            self._write_cesar_input(cesar_input, ref_seqs, query_seqs,
                                    splice_sites, num_exons, locus_strand)


            saved_in = self._save_debug(cesar_input, '.input.fa')
            if self.verbose:
                print(f'  Saved CESAR input  → {saved_in}')

            self._run_cesar_binary(cesar_input, cesar_output)

            with open(cesar_output) as f:
                raw_lines = [l.rstrip('\n') for l in f]

            saved_raw = self._save_debug(cesar_output, '.output_raw.alignment')
            if self.verbose:
                print(f'  Saved CESAR raw output → {saved_raw}')

            if False: # self.run_post_process disabled for debugging:
                processor = CesarOutputPostProcessor(verbose=False)
                temp_out  = cesar_output + '.tmp'
                processor.process_file(cesar_output, temp_out)
                os.replace(temp_out, cesar_output)
                saved_post = self._save_debug(cesar_output, '.output_post.alignment')
                if self.verbose:
                    print(f'  Saved CESAR post output → {saved_post}')

            if self.verbose:
                print(f'  CESAR output lines: {len(raw_lines)}')
                for i, l in enumerate(raw_lines[:4]):
                    print(f'  line[{i}]: {l[:120]}')

            # Robust header parsing immune to prepended CESAR 3 warning lines
            ref_idx = next((i for i, l in enumerate(raw_lines) if l.startswith('>')), None)
            if ref_idx is not None and len(raw_lines) >= ref_idx + 4:
                result = {
                    'ref_header'  : raw_lines[ref_idx],
                    'ref_seq'     : raw_lines[ref_idx + 1],
                    'query_header': raw_lines[ref_idx + 2],
                    'query_seq'   : raw_lines[ref_idx + 3],
                }

        except Exception as e:
            if self.verbose:
                print(f'  CESAR run failed: {e}')
            result = None

        finally:
            for f in (cesar_input, cesar_output):
                if os.path.exists(f):
                    os.remove(f)

        return result

    # ── input writer ──────────────────────────────────────────────────────────

    def _write_cesar_input(self, output_file, ref_seqs, query_seqs,
                           splice_sites, num_exons, target_strand):
        with open(output_file, 'w') as f:

            for exon_id, sequence in ref_seqs.items():
                exon_num = int(exon_id.split('_')[-1])

                acc_profile = os.path.join('extra', 'tables', self.clade, 'acc_profile.txt')
                do_profile  = os.path.join('extra', 'tables', self.clade, 'do_profile.txt')

                if exon_num == 0:
                    acc_profile = os.path.join('extra', 'tables', self.clade,
                                               'firstCodon_profile.txt')

                if exon_num == num_exons - 1:
                    do_profile = os.path.join('extra', 'tables', self.clade,
                                              'lastCodon_profile.txt')

                if splice_sites:
                    if (splice_sites['acc'].get(exon_id) == 'AC' and exon_num != 0):
                        acc_profile = os.path.join('extra', 'tables', self.clade,
                                                   'u12_acc_profile.txt')
                    if (splice_sites['donor'].get(exon_id) == 'AT' and exon_num != num_exons - 1):
                        do_profile = os.path.join('extra', 'tables', self.clade,
                                                  'u12_donor_profile.txt')

                f.write(f'>{self.transcript_id}_{exon_num}\t{acc_profile}\t{do_profile}\n')
                f.write(f'{sequence}\n')

            f.write('####\n')

            for header, sequence in query_seqs.items():
                _, _, _, query_strand, _ = header.split('#')
                
                # Flip ONLY if the query MAF strand doesn't match its OWN locus strand
                if query_strand != target_strand:
                    sequence = utils.reverse_comp(sequence)
                    
                f.write(f'>{header}\n')
                f.write(f'{sequence}\n')

    # ── binary runner ─────────────────────────────────────────────────────────

    def _run_cesar_binary(self, cesar_input_file, output_fasta):
        original_dir = os.getcwd()
        os.chdir(self.cesar_dir)
        try:
            # Auto-detect binary name for CESAR 2.0 vs CESAR 3
            cesar_binary = os.path.join(self.cesar_dir, 'cesar3')
            if not os.path.exists(cesar_binary):
                cesar_binary = os.path.join(self.cesar_dir, 'cesar')
            if not os.path.exists(cesar_binary):
                raise FileNotFoundError(f"CRITICAL: Neither 'cesar' nor 'cesar3' binary found in {self.cesar_dir}")
                
            cmd = f'{cesar_binary} {cesar_input_file}'
            result       = utils.run_subprocess(cmd, return_output=True)
            with open(output_fasta, 'w') as f:
                f.write(result)
            return result
        finally:
            os.chdir(original_dir)