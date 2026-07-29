import os
import utils

class ReferenceExon:
    """
    Extracts and formats reference multi-exon sequences for CESAR 2.0.
    
    - Uses a single BED file extraction for high-speed I/O.
    - Calculates biological exon phases to preserve reading frames.
    - Formats sequences with '|' markers and lowercasing as required by CESAR.
    """
    def __init__(self, exons_list, chromosome, strand, two_bit_file):
        self.exons_list   = exons_list
        self.chromosome   = chromosome
        self.strand       = strand
        self.two_bit_file = two_bit_file

    def extract_for_cesar(self):
        """
        Main wrapper called by cesar_runner.py.
        Returns: (formatted_seqs_dict, splice_sites_dict)
        """
        # 1. Extract raw sequences + 2bp splice sites via optimized BED method
        raw_seqs_dict = self._get_exon_sequences_via_bed()
        
        # 2. Extract the dinucleotide splice sites (donor/acceptor)
        splice_sites  = self._extract_splice_sites(raw_seqs_dict)
        
        # 3. Calculate biological reading frame phases
        exon_phases   = self._get_exon_phases()
        
        # 4. Strip splice sites and format the sequence text with phase markers
        formatted_seqs = self._format_multi_exon(raw_seqs_dict, exon_phases)
        
        return formatted_seqs, splice_sites

    def _get_exon_sequences_via_bed(self):
        """
        Writes all exon coordinates to a single BED file in RAM and extracts 
        them in one subprocess call.
        """
        temp_bed   = utils.create_temp_devshm()
        temp_fasta = utils.create_temp_devshm()

        with open(temp_bed, 'w') as FO:
            for exon_ct, exon in enumerate(self.exons_list):
                start, stop = exon.split('-')
                # Expand by 2bp for splice sites, preventing negative coordinates
                start = max(0, int(start) - 2)
                stop  = int(stop) + 2
                FO.write(f"{self.chromosome}\t{start}\t{stop}\texon_{exon_ct}\n")

        cmd = f"twoBitToFa {self.two_bit_file} {temp_fasta} -bed={temp_bed}"
        utils.run_subprocess(cmd)
        
        seqs_dict = utils.fasta_to_dict(temp_fasta)

        if self.strand == '-':
            seqs_dict = {k: utils.reverse_comp(v) for k, v in seqs_dict.items()}

        if os.path.exists(temp_bed):   os.remove(temp_bed)
        if os.path.exists(temp_fasta): os.remove(temp_fasta)
        
        return seqs_dict

    def _extract_splice_sites(self, seqs_dict):
        """
        Returns a dictionary of acceptor (first 2bp) and donor (last 2bp) sites.
        """
        ss = {"acc": {}, "donor": {}}
        for exon_id, seq in seqs_dict.items():
            ss["acc"][exon_id]   = seq[0:2]
            ss["donor"][exon_id] = seq[-2:]
        return ss

    def _get_exon_phases(self):
        """
        Calculates how introns interrupt the 3-letter codon reading frame.
        """
        length_cum = 0
        exons_length = []
        for exon in self.exons_list:
            s, t = exon.split('-')
            length_cum += abs(int(t) - int(s))
            exons_length.append(length_cum)
            
        phases = []
        for i, cum in enumerate(exons_length):
            p5 = 0 if i == 0 else 3 - exons_length[i-1] % 3
            p3 = cum % 3
            p5 = 0 if p5 == 3 else p5
            p3 = 0 if p3 == 3 else p3
            phases.append(f"{p5}-{p3}")
            
        return phases

    def _format_multi_exon(self, seqs_dict, phases):
        """
        Strips the 2bp splice sites and inserts '|' boundaries based on the 
        phase so CESAR knows exactly where the reading frame starts and stops.
        """
        formatted = {}
        # Ensure strict genomic order using the numeric suffix of 'exon_X'
        for i, exon_id in enumerate(sorted(seqs_dict, key=lambda x: int(x.split('_')[1]))):
            
            # Strip the 2bp outer splice sites, leaving only the coding sequence
            seq = seqs_dict[exon_id][2:-2].upper()
            p5, p3 = map(int, phases[i].split('-'))
            
            # Format 5' end
            seq = ('|' + seq) if p5 == 0 else (seq[:p5].lower() + '|' + seq[p5:])
            
            # Format 3' end
            seq = (seq + '|') if p3 == 0 else (seq[:-p3] + '|' + seq[-p3:].lower())
            
            formatted[exon_id] = seq
            
        return formatted