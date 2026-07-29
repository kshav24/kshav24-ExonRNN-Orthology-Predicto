#!/usr/bin/env python3
"""
CesarOutputPostProcessor - Clean CESAR output (both modes)

Removes structural elements from CESAR output:
- >>> markers (19 characters) from multi-exon
- ALL spaces from both single-exon and multi-exon
- Corresponding gaps in query sequence

This makes alignments cleaner and easier to read!
"""

class CesarOutputPostProcessor:
    """
    Post-process CESAR output to remove spaces and exon separators

    Works for BOTH single-exon and multi-exon modes:
    - Removes >>> markers (multi-exon only)
    - Removes ALL spaces (both modes)
    - Removes corresponding positions in query
    """

    def __init__(self, verbose=False):
        """
        Initialize post-processor

        Args:
            verbose: Print processing details
        """
        self.verbose = verbose

    def process_file(self, input_fasta, output_fasta):
        """
        Process CESAR output file (any mode)

        Args:
            input_fasta: Path to CESAR output
            output_fasta: Path to write cleaned output

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read input file
            with open(input_fasta, 'r') as f:
                lines = [line.rstrip('\n') for line in f]

            if len(lines) < 4:
                if self.verbose:
                    print(f"  ERROR: Invalid FASTA format (only {len(lines)} lines)")
                return False

            # Parse FASTA (assumes 4 lines)
            ref_header = lines[0]
            ref_seq = lines[1]
            query_header = lines[2]
            query_seq = lines[3]

            if self.verbose:
                print(f"  Input ref length:   {len(ref_seq)}")
                print(f"  Input query length: {len(query_seq)}")

            # Clean sequences
            cleaned_ref, cleaned_query = self._clean_alignment(ref_seq, query_seq)

            if self.verbose:
                print(f"  Output ref length:   {len(cleaned_ref)}")
                print(f"  Output query length: {len(cleaned_query)}")

            # Write output
            with open(output_fasta, 'w') as f:
                f.write(f"{ref_header}\n")
                f.write(f"{cleaned_ref}\n")
                f.write(f"{query_header}\n")
                f.write(f"{cleaned_query}\n")

            return True

        except Exception as e:
            if self.verbose:
                print(f"  ERROR: {e}")
            return False

    def _clean_alignment(self, ref_seq, query_seq):
        """
        Remove >>> markers and ALL spaces from alignment

        Algorithm:
        1. Scan reference sequence character by character
        2. Find >>> markers (19 chars) - skip them
        3. Find spaces - skip them
        4. Keep all other positions
        5. Apply same filtering to query

        Args:
            ref_seq: Reference sequence with possible >>> and spaces
            query_seq: Query sequence with corresponding gaps

        Returns:
            (cleaned_ref, cleaned_query) tuple
        """
        if len(ref_seq) != len(query_seq):
            raise ValueError(
                f"Sequence length mismatch: ref={len(ref_seq)}, query={len(query_seq)}"
            )

        # Track positions to keep
        keep_positions = []

        i = 0
        markers_removed = 0
        spaces_removed = 0

        while i < len(ref_seq):
            ref_char = ref_seq[i]

            # Check for >>> marker (19 characters)
            if ref_char == '>' and i + 18 < len(ref_seq):
                if ref_seq[i:i+19] == '>'*19:
                    markers_removed += 1
                    i += 19
                    continue

            # Check for space
            if ref_char == ' ':
                spaces_removed += 1
                i += 1
                continue

            # Keep this position
            keep_positions.append(i)
            i += 1

        # Build cleaned sequences
        cleaned_ref = ''.join(ref_seq[pos] for pos in keep_positions)
        cleaned_query = ''.join(query_seq[pos] for pos in keep_positions)

        if self.verbose:
            if markers_removed > 0:
                print(f"  Removed {markers_removed} >>> markers")
            if spaces_removed > 0:
                print(f"  Removed {spaces_removed} spaces")

        return cleaned_ref, cleaned_query


def main():
    """Test the post-processor"""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python cesarPostProcessor.py input.fa output.fa")
        print()
        print("Works for both single-exon and multi-exon CESAR output!")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    processor = CesarOutputPostProcessor(verbose=True)

    print(f"Processing: {input_file}")
    success = processor.process_file(input_file, output_file)

    if success:
        print(f"✓ Successfully processed -> {output_file}")
    else:
        print(f"✗ Failed to process {input_file}")
        sys.exit(1)


if __name__ == '__main__':
    main()
