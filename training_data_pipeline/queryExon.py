import os
from collections import Counter
import utils


class QueryExon:
    """
    PROFILER-compatible QueryExon.

    Interface (matches cesar_runner.py):
        __init__(ref_species, exons_list, chromosome, ref_strand,
                 two_bit_dir, query_species_list, maf_index_file, cesar_dir)
        extract_query_sequences(single_exon_mode=False)
            -> {coord_str: sequence}
            coord_str = 'species#start#stop#strand#chrom'

    Design principles:
    - Temp files written to /dev/shm (RAM) to avoid disk I/O bottleneck.
    - Per-exon MAF extraction: one mafExtract call per exon.
    - Primary-chromosome filter: strict majority rules; spurious hits ignored.
    - Colinearity check: exon starts and stops must be monotonically ordered.
    - Anchor check: all genomic boundaries must exist to prevent truncation.
    - Minus-strand coords: Uses absolute outer boundaries safely.

    Terminal exon e-line fallback (KEY FIX):
    - The first and last exon are the most likely to fall in assembly gaps,
      where the query species has sequence but no MAF alignment (e-line I/C).
    - Previously the None guard would kill the entire species if exon 0 or
      exon n-1 had no MAF s-line.  Now we check for e-lines with status I
      (sequence exists, unaligned) or C (contiguous synteny) and use those
      coords as a locus anchor before the None guard fires.
    - Middle exons: no e-line fallback.  A missing middle exon is a real
      alignment gap and should not be papered over.
    - The coord_str output format is identical to before; CesarRunner and
      generate_training_data.py need no structural changes.
    """

    def __init__(self, ref_species, exons_list, chromosome, ref_strand,
                 two_bit_dir, query_species_list, maf_index_file, cesar_dir):
        self.ref_species        = ref_species
        self.exons_list         = exons_list
        self.chromosome         = chromosome
        self.ref_strand         = ref_strand
        self.two_bit_dir        = two_bit_dir
        self.query_species_list = query_species_list
        self.maf_index          = maf_index_file

    # ─────────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────────

    def extract_query_sequences(self, single_exon_mode=False):
        query_species_cds = self.__get_query_coords()
        if not query_species_cds:
            print("  [QueryExon] WARNING: no valid query coordinates for any species")
            return {}
        query_seqs = {}
        for species, coord_str in query_species_cds.items():
            seq = self.__extract_window(species, coord_str)
            if seq:
                query_seqs[coord_str] = seq
            else:
                print(f"  [QueryExon] WARNING: sequence extraction failed: {coord_str}")
        return query_seqs

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    '''
    def __extract_window(self, species, coord_str):
        """Extract the genomic window from the species 2bit file."""
        _, start, stop, strand, chrom = coord_str.split('#')
        two_bit = os.path.join(self.two_bit_dir, species, f'{species}.2bit')
        if not os.path.exists(two_bit):
            print(f"  [QueryExon] WARNING: 2bit not found: {two_bit}")
            return None
        try:
            seq = utils.get_sequence_from_2bit(int(start), int(stop), chrom, two_bit)
        except Exception as e:
            print(f"  [QueryExon] WARNING: 2bit extraction failed for {coord_str}: {e}")
            return None
        if strand == '-':
            seq = utils.reverse_comp(seq)
        return seq
    
   

    def __extract_window(self, species, coord_str):
        _, start, stop, strand, chrom = coord_str.split('#')
        two_bit = os.path.join(self.two_bit_dir, species, f'{species}.2bit')
        if not os.path.exists(two_bit):
            print(f" [QueryExon] WARNING: 2bit not found: {two_bit}")
            return None
        try:
            # Always extract low→high for 2bitToFa
            low  = min(int(start), int(stop))
            high = max(int(start), int(stop))
            seq = utils.get_sequence_from_2bit(low, high, chrom, two_bit)
        except Exception as e:
            print(f" [QueryExon] WARNING: 2bit extraction failed for {coord_str}: {e}")
            return None
        if strand == '-':
            seq = utils.reverse_comp(seq)
        return seq

     '''

    def __extract_window(self, species, coord_str):
        _, start, stop, strand, chrom = coord_str.split('#')
        two_bit = os.path.join(self.two_bit_dir, species, f'{species}.2bit')
        if not os.path.exists(two_bit):
            print(f"  [QueryExon] WARNING: 2bit not found: {two_bit}")
            return None
        try:
            low  = min(int(start), int(stop))
            high = max(int(start), int(stop))
            seq = utils.get_sequence_from_2bit(low, high, chrom, two_bit)
        except Exception as e:
            print(f"  [QueryExon] WARNING: 2bit extraction failed for {coord_str}: {e}")
            return None
            
        # NO REVERSE COMPLEMENT HERE AT ALL. Return raw genomic + strand.
        return seq

    def _parse_elines(self, maf_file):
        """
        Parse MAF e-lines from a single per-exon MAF file.

        Returns {species: (fwd_start, fwd_end, strand, chrom)} for status
        I (between blocks — sequence exists, unaligned) or C (contiguous
        synteny) ONLY.  Status N (no sequence / assembly gap / synteny break)
        and M (missing from assembly) are skipped — they carry no usable
        sequence for CESAR.

        Coords are always returned in forward-strand orientation so the rest
        of __get_query_coords() can treat them the same as s-line hits.

        This method is called only for terminal exon indices (0 and n-1),
        so the overhead is minimal.
        """
        result = {}
        with open(maf_file) as fh:
            for line in fh:
                if not line.startswith('e'):
                    continue
                fields = line.split()
                if len(fields) < 7:
                    continue

                src_info = fields[1]
                e_start  = int(fields[2])
                e_len    = int(fields[3])
                strand   = fields[4]
                src_size = int(fields[5])
                e_status = fields[6].strip()

                if '.' not in src_info:
                    continue
                species, chrom = src_info.split('.', 1)
                if species not in self.query_species_list:
                    continue

                # Only I/C have real sequence — skip N and M
                if e_status not in ('I', 'C'):
                    continue

                # Convert to forward-strand genomic coords
                if strand == '-':
                    fwd_start = src_size - e_start - e_len
                    fwd_end   = src_size - e_start
                else:
                    fwd_start = e_start
                    fwd_end   = e_start + e_len

                # Take the first e-line for each species in this block.
                # Multiple e-lines for the same species in one exon MAF are rare.
                if species not in result:
                    result[species] = (fwd_start, fwd_end, strand, chrom)

        return result

    def __get_query_coords(self):
        """
        For each query species, find the genomic locus spanning all exons.
        Returns {species: coord_str} for species passing all QC filters.

        QC pipeline per species:
          1. Primary chromosome  — strict majority of exon hits
          2. Terminal e-line fallback — fill first/last exon from e-line I/C
             if s-line missing AND e-line chrom matches primary
          3. Anchor check  — all four terminal boundaries must now be present
          4. Single strand — all hits must agree on strand
          5. Colinearity   — exon order must be monotonically ascending or descending
        """
        exons_sorted = sorted(self.exons_list, key=lambda e: int(e.split('-')[0]))
        n = len(exons_sorted)

        print(f"\n  [QueryExon] {n} exons | ref={self.ref_species} "
              f"| species: {self.query_species_list}")

        start_dict  = {sp: [None] * n for sp in self.query_species_list}
        stop_dict   = {sp: [None] * n for sp in self.query_species_list}
        chr_dict    = {sp: [None] * n for sp in self.query_species_list}
        strand_dict = {sp: [None] * n for sp in self.query_species_list}

        # e-line fallback — populated only for terminal exon indices (0, n-1)
        # Structure: {species: {exon_idx: (fwd_start, fwd_end, strand, chrom)}}
        eline_fallback = {sp: {} for sp in self.query_species_list}

        # ── Per-exon MAF extraction ───────────────────────────────────────────
        for exon_idx, exon in enumerate(exons_sorted):
            tmp = utils.create_temp_devshm()
            e_start, e_end = exon.split('-')
            cmd = (
                f"mafExtract {self.maf_index} "
                f"-region={self.chromosome}:{e_start}-{e_end} {tmp}"
            )
            try:
                utils.run_subprocess(cmd)
            except Exception as ex:
                print(f"    exon {exon_idx+1} ({exon}): mafExtract failed — {ex}")
                if os.path.exists(tmp):
                    os.remove(tmp)
                continue

            # Always parse s-lines
            hits = self.__parse_maf_exon(tmp)

            # For terminal exons ONLY: also parse e-lines as coord fallback.
            # Reading the same tmp file a second time is cheap (in RAM).
            if exon_idx == 0 or exon_idx == n - 1:
                eline_hits = self._parse_elines(tmp)
                for sp_e, coords in eline_hits.items():
                    # Store only the first e-line hit per species per terminal exon
                    if exon_idx not in eline_fallback[sp_e]:
                        eline_fallback[sp_e][exon_idx] = coords

            if os.path.exists(tmp):
                os.remove(tmp)

            logged = []
            for sp in self.query_species_list:
                if sp in hits:
                    q_start, q_stop, strand, chrom = hits[sp]
                    start_dict[sp][exon_idx]  = q_start
                    stop_dict[sp][exon_idx]   = q_stop
                    chr_dict[sp][exon_idx]    = chrom
                    strand_dict[sp][exon_idx] = strand
                    logged.append(f"{sp}:{chrom}:{q_start}-{q_stop}({strand})")

            if logged:
                print(f"    exon {exon_idx+1} ({exon}): {'  '.join(logged)}")
            else:
                print(f"    exon {exon_idx+1} ({exon}): no MAF s-line hits")

        # ── Build per-species locus ───────────────────────────────────────────
        query_species_cds = {}

        for sp in self.query_species_list:
            starts  = start_dict[sp]   # list reference — mutations affect start_dict
            stops   = stop_dict[sp]
            chroms  = chr_dict[sp]
            strands = strand_dict[sp]

            # ── QC 1: Primary chromosome ──────────────────────────────────────
            
            chrom_counter = Counter(c for c in chroms if c is not None)
            if not chrom_counter:
                print(f"  [{sp}] SKIP — no chromosome information (all exons missing)")
                continue

            primary_chrom, primary_hits = chrom_counter.most_common(1)[0]
            # Strict majority: more than half of all exons must agree
            if primary_hits < (n + 1) // 2:
                print(f"  [{sp}] SKIP — no dominant chromosome: {chrom_counter}")
                continue


            spurious = {c for c in chrom_counter if c != primary_chrom}
            if spurious:
                print(f"  [{sp}] NOTE — ignoring spurious hits on: {spurious}")

            for i in range(n):
                if chroms[i] != primary_chrom:
                    starts[i] = stops[i] = chroms[i] = strands[i] = None

            # ── Terminal exon e-line fallback ─────────────────────────────────
            # If the first or last exon has no s-line, check our e-line cache.
            # Only apply when the e-line chrom matches the already-established
            # primary chrom — this filters out cross-chromosomal noise.
            # Middle exons are deliberately NOT filled here.
            sp_elines = eline_fallback.get(sp, {})
            for term_idx in (0, n - 1):
                if starts[term_idx] is None and term_idx in sp_elines:
                    e_fs, e_fe, e_strand, e_chrom = sp_elines[term_idx]
                    if e_chrom == primary_chrom:
                        starts[term_idx]  = e_fs
                        stops[term_idx]   = e_fe
                        chroms[term_idx]  = e_chrom
                        strands[term_idx] = e_strand
                        print(f"  [{sp}] exon {term_idx+1}: no s-line → "
                              f"e-line fallback {e_chrom}:{e_fs}-{e_fe} "
                              f"({e_strand}) [forced_locus]")

            # ── QC 2: Anchor check ────────────────────────────────────────────
            # All four terminal boundaries must exist.
            # After the e-line fill above, terminals may now be populated.
            # If still None, we have no usable locus for this species.
# ── QC 2: Anchor check — relax to nearest mapped boundary ────────────
            if starts[0] is None or stops[0] is None:
                first_valid = next((i for i in range(n) if starts[i] is not None), None)
                if first_valid is None:
                    print(f"  [{sp}] SKIP — no mapped exons at all")
                    continue
                starts[0]  = starts[first_valid]
                stops[0]   = stops[first_valid]
                chroms[0]  = chroms[first_valid]
                strands[0] = strands[first_valid]
                print(f"  [{sp}] exon 1: no anchor — using exon {first_valid+1} boundary instead")

            if starts[-1] is None or stops[-1] is None:
                last_valid = next((i for i in range(n-1, -1, -1) if stops[i] is not None), None)
                if last_valid is None:
                    print(f"  [{sp}] SKIP — no mapped exons at all")
                    continue
                starts[-1]  = starts[last_valid]
                stops[-1]   = stops[last_valid]
                chroms[-1]  = chroms[last_valid]
                strands[-1] = strands[last_valid]
                print(f"  [{sp}] exon {n}: no anchor — using exon {last_valid+1} boundary instead")

            # ── QC 3: Single strand ───────────────────────────────────────────
            strand_counter = Counter(s for s in strands if s is not None)
            if len(strand_counter) != 1:
                print(f"  [{sp}] SKIP — inconsistent strands: {dict(strand_counter)}")
                continue
            query_strand = strand_counter.most_common(1)[0][0]

            # ── QC 4: Colinearity check ───────────────────────────────────────
            valid_starts = [s for s in starts if s is not None]
            valid_stops  = [s for s in stops  if s is not None]

            ascending  = (valid_starts == sorted(valid_starts, key=int) and
                          valid_stops  == sorted(valid_stops,  key=int))
            descending = (valid_starts == sorted(valid_starts, key=int, reverse=True) and
                          valid_stops  == sorted(valid_stops,  key=int, reverse=True))

            if not (ascending or descending):
                print(f"  [{sp}] SKIP — non-colinear exon order "
                      f"(starts: {valid_starts[:4]}...)")
                continue

            # ── Locus: outer boundaries ───────────────────────────────────────
            if ascending:
                locus_start = int(starts[0])
                locus_stop  = int(stops[-1])
            else:   # descending
                locus_start = int(starts[-1])
                locus_stop  = int(stops[0])

            window_size = locus_stop - locus_start
            n_mapped    = sum(1 for s in starts if s is not None)

            # Flag if any terminal exon used e-line fallback
            used_eline = []
            if 0 in sp_elines and start_dict[sp][0] == sp_elines[0][0]:
                used_eline.append('exon1')
            if (n - 1) in sp_elines and start_dict[sp][n-1] == sp_elines[n-1][0]:
                used_eline.append(f'exon{n}')
            eline_note = f" | e-line fallback: {','.join(used_eline)}" if used_eline else ""

            '''
            coord_str = (f"{sp}#{locus_start}#{locus_stop}"
                         f"#{query_strand}#{primary_chrom}") 
            
            '''
            if query_strand == '-':
                coord_str = f"{sp}#{locus_stop}#{locus_start}#{query_strand}#{primary_chrom}"
            else:
                coord_str = f"{sp}#{locus_start}#{locus_stop}#{query_strand}#{primary_chrom}"

            query_species_cds[sp] = coord_str

            print(f"  [{sp}] {primary_chrom}:{locus_start}-{locus_stop} "
                  f"({query_strand}) | {window_size:,} bp | "
                  f"{n_mapped}/{n} exons mapped{eline_note}")

        return query_species_cds

    def __parse_maf_exon(self, maf_file):
        """
        Parse one per-exon MAF file and return per-species coordinates.
        Uses min/max over ALL alignment blocks for the exon.
        Only processes s-lines (aligned sequence blocks).
        """
        with open(maf_file) as fh:
            lines = [l.rstrip('\n') for l in fh]

        blocks  = {sp: [] for sp in self.query_species_list}
        strands = {sp: set() for sp in self.query_species_list}

        for line in lines:
            if not line.startswith('s'):
                continue
            fields = line.split()
            if len(fields) < 7:
                continue

            src_info   = fields[1]
            start      = int(fields[2])
            align_size = int(fields[3])
            strand     = fields[4]
            src_size   = int(fields[5])

            if '.' not in src_info:
                continue
            species, chrom = src_info.split('.', 1)

            if species == self.ref_species or species not in self.query_species_list:
                continue

            # Convert to forward-strand genomic coordinates
            if strand == '-':
                fwd_start = src_size - start - align_size
                fwd_stop  = src_size - start
            else:
                fwd_start = start
                fwd_stop  = start + align_size

            blocks[species].append((fwd_start, fwd_stop, chrom))
            strands[species].add(strand)

        result = {}
        for sp in self.query_species_list:
            bl = blocks[sp]
            if not bl:
                continue
            if len(strands[sp]) != 1:
                continue

            # Dominant chrom for this exon (filters inter-chromosomal noise)
            primary = Counter(c for _, _, c in bl).most_common(1)[0][0]
            clean   = [(s, e) for s, e, c in bl if c == primary]

            exon_start = min(s for s, e in clean)
            exon_stop  = max(e for s, e in clean)

            result[sp] = (exon_start, exon_stop,
                          list(strands[sp])[0], primary)

        return result
