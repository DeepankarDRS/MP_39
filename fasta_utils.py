# fasta_utils.py
from typing import Iterator, Tuple, List

def parse_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """
    Yield (header, sequence) pairs from a FASTA file.
    Header excludes the leading '>'.
    """
    header = None
    seq_chunks: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)

def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return 100.0 * gc / len(seq) if seq else 0.0

def write_fasta(records: Iterator[Tuple[str, str]], out_path: str, line_width: int = 60):
    with open(out_path, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                out.write(seq[i:i+line_width] + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse FASTA and print summary")
    parser.add_argument("fasta", help="input FASTA file")
    parser.add_argument("--min-length", type=int, default=0, help="filter sequences shorter than this")
    parser.add_argument("--out", help="optional output FASTA for filtered sequences")
    args = parser.parse_args()

    filtered = []
    for header, seq in parse_fasta(args.fasta):
        length = len(seq)
        gc = gc_content(seq)
        print(f"{header}\tlen={length}\tGC={gc:.2f}%")
        if length >= args.min_length:
            filtered.append((header, seq))

    if args.out:
        write_fasta(iter(filtered), args.out)
        print(f"Wrote {len(filtered)} records to {args.out}")
