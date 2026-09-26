#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass
class Sample:
    sid: int
    task: str
    species: str
    chrom: str
    start: int
    end: int
    bed_fields: list[str]
    csv_fields: dict[str, str] | None
    gene: str
    locus: str
    source_bed: Path
    source_csv: Path | None
    row_index: int
    block: tuple[str, str, int]
    group: tuple[str, str, int] | None = None
    split: str = ""
    removed: str = ""

    @property
    def sequence(self) -> str:
        return (self.csv_fields or {}).get("sequence", "").upper().strip()

    @property
    def label(self) -> str:
        return (self.csv_fields or {}).get("label", "unknown").strip()


class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        if key not in self.parent:
            self.parent[key] = key
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def path_from(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_samples(args):
    root = Path(args.input_root).resolve()
    chrom_species = {}
    if args.chrom_species_map:
        with args.chrom_species_map.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                chrom, species_name = row["chrom"].strip(), row["species"].strip()
                if chrom in chrom_species and chrom_species[chrom] != species_name:
                    raise ValueError(f"Conflicting species for {chrom}")
                chrom_species[chrom] = species_name
    specs = list(csv.DictReader(open(args.task_map, newline="", encoding="utf-8"), delimiter="\t"))
    if not specs or not {"task", "species", "bed"}.issubset(specs[0]):
        raise ValueError("task map requires task, species, bed columns")
    samples, schemas, task_sources = [], {}, {}
    saw_csv = saw_bed_only = False
    for spec in specs:
        task, species = spec["task"].strip(), spec["species"].strip()
        if not task or "/" in task or ".." in task:
            raise ValueError(f"Invalid task/species: {spec}")
        bed = path_from(root, spec["bed"].strip()).resolve()
        csv_value = (spec.get("csv") or "").strip()
        source_csv = path_from(root, csv_value).resolve() if csv_value else None
        if not bed.is_file() or (source_csv and not source_csv.is_file()):
            raise FileNotFoundError(f"Missing BED/CSV for {task}, {species}: {bed}, {source_csv}")
        source_key = (task, species, bed)
        if source_key in task_sources:
            raise ValueError(f"Repeated task-map entry: {source_key}")
        task_sources[source_key] = source_csv
        gene_col = int(spec["gene_col"]) - 1 if (spec.get("gene_col") or "").strip() else None
        locus_col = int(spec["locus_col"]) - 1 if (spec.get("locus_col") or "").strip() else None
        species_col = int(spec["species_col"]) - 1 if (spec.get("species_col") or "").strip() else None
        csv_rows = None
        if source_csv:
            saw_csv = True
            if not args.confirm_row_alignment:
                raise ValueError("CSV/BED row matching cannot be inferred. Verify identical order and pass --confirm-row-alignment")
            with source_csv.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "sequence" not in reader.fieldnames:
                    raise ValueError(f"{source_csv}: missing sequence column")
                schemas[source_csv] = list(reader.fieldnames)
                csv_rows = list(reader)
        else:
            saw_bed_only = True
        n = 0
        with bed.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 3:
                    raise ValueError(f"{bed}:{line_no}: expected BED3+ tab-separated line")
                chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                if start < 0 or end <= start or end - start > args.max_interval_bp:
                    raise ValueError(f"{bed}:{line_no}: invalid/too long interval {start}-{end}")
                if csv_rows is not None and n >= len(csv_rows):
                    raise ValueError(f"{bed}: more BED than CSV rows")
                for col in (gene_col, locus_col, species_col):
                    if col is not None and (col < 3 or col >= len(fields)):
                        raise ValueError(f"{bed}:{line_no}: gene/locus/species column outside BED extra columns")
                row_species = fields[species_col].strip() if species_col is not None else (species or chrom_species.get(chrom, ""))
                if not row_species:
                    raise ValueError(f"{bed}:{line_no}: species unknown; set species, species_col or --chrom-species-map")
                if species and chrom in chrom_species and chrom_species[chrom] != species:
                    raise ValueError(f"{bed}:{line_no}: task species disagrees with chromosome map")
                row = csv_rows[n] if csv_rows is not None else None
                if row and not row["sequence"].strip():
                    raise ValueError(f"{source_csv} row {n+2}: empty sequence")
                # A stable namespace prevents different species/assemblies sharing coordinate IDs.
                block = (row_species, chrom, ((start + end - 1) // 2) // args.block_bp)
                samples.append(Sample(len(samples), task, row_species, chrom, start, end, fields,
                                      row, fields[gene_col] if gene_col is not None else "",
                                      fields[locus_col] if locus_col is not None else "",
                                      bed, source_csv, n, block))
                n += 1
        if csv_rows is not None and n != len(csv_rows):
            raise ValueError(f"{bed}: {n} BED rows but {source_csv}: {len(csv_rows)} CSV rows")
    if not samples:
        raise ValueError("No BED rows found")
    if args.similarity == "mmseqs" and saw_bed_only:
        raise ValueError("--similarity mmseqs requires a sequence CSV for EVERY task-map entry")
    if args.similarity == "mmseqs" and not saw_csv:
        raise ValueError("--similarity mmseqs requires CSV sequences")
    return samples, schemas, task_sources


def make_groups(samples):
    dsu, ties = DSU(), {}
    for s in samples:
        dsu.find(s.block)
        for prefix, value in (("gene", s.gene), ("locus", s.locus)):
            if not value or value in {".", "NA", "nan"}:
                continue
            key = ((s.species, "gene", value) if prefix == "gene"
                   else (s.species, s.chrom, "locus", value))
            if key in ties:
                dsu.union(s.block, ties[key])
            else:
                ties[key] = s.block
    groups = defaultdict(list)
    for s in samples:
        s.group = dsu.find(s.block)
        groups[s.group].append(s)
    return groups


def dimensions(s: Sample):
    # Independent task-label and task-species balance, plus total per task.
    d = [("task", s.task)]
    if s.csv_fields is not None and "label" in s.csv_fields:
        d.append(("label", s.task, s.label))
    if s.task.startswith("splice"):
        d.append(("species", s.task, s.species))
    return d


def assign(groups, seed, tries):
    vectors = {g: Counter(d for s in rows for d in dimensions(s)) for g, rows in groups.items()}
    total = sum(vectors.values(), Counter())
    keys = list(groups)
    best_cost, best = float("inf"), None
    for attempt in range(tries):
        rng = random.Random(seed + 1009 * attempt)
        rng.shuffle(keys)
        # Size ordering limits the damage caused by placing a large component last.
        keys.sort(key=lambda g: -len(groups[g]))
        counts = {split: Counter() for split in SPLITS}
        assignment = {}
        for g in keys:
            choices = list(SPLITS)
            rng.shuffle(choices)
            def delta(split):
                score = 0.0
                for dim, inc in vectors[g].items():
                    target = RATIOS[split] * total[dim]
                    old = counts[split][dim] - target
                    score += ((old + inc) ** 2 - old ** 2) / max(total[dim], 10) ** 2
                return score
            chosen = min(choices, key=delta)
            assignment[g] = chosen
            counts[chosen].update(vectors[g])
        cost = sum(
            (counts[split][dim] / max(n, 1) - RATIOS[split]) ** 2
            for dim, n in total.items() for split in SPLITS
        )
        if cost < best_cost:
            best_cost, best = cost, assignment.copy()
    for g, rows in groups.items():
        for s in rows:
            s.split = best[g]
    return best_cost


def purge_boundaries(samples, buffer_bp, block_bp):
    block_splits = {}
    for s in samples:
        if s.block in block_splits and block_splits[s.block] != s.split:
            raise AssertionError("A genomic block spans two splits")
        block_splits[s.block] = s.split
    for s in samples:
        idx = s.block[2]
        lo_boundary = idx * block_bp
        hi_boundary = (idx + 1) * block_bp
        before = (s.species, s.chrom, idx - 1)
        after = (s.species, s.chrom, idx + 1)
        # Interval-to-boundary distance, not midpoint-to-boundary distance.
        if (before in block_splits and block_splits[before] != s.split
                and s.start < lo_boundary + buffer_bp):
            s.removed = "cross_split_boundary"
        if (after in block_splits and block_splits[after] != s.split
                and s.end > hi_boundary - buffer_bp):
            s.removed = "cross_split_boundary"
        # For a BED window longer than the distance to its anchor's next border,
        # the above tests also catch it when the neighboring block has samples.
    return block_splits


def canonical(seq):
    rc = seq.translate(RC)[::-1]
    return min(seq, rc)


def purge_exact_duplicates(samples):
    by_seq = defaultdict(list)
    for s in samples:
        if not s.removed and s.sequence:
            by_seq[hashlib.sha256(canonical(s.sequence).encode()).hexdigest()].append(s)
    for rows in by_seq.values():
        splits = {s.split for s in rows}
        if len(splits) > 1:
            priority = "train" if "train" in splits else "test"
            for s in rows:
                if s.split != priority:
                    s.removed = "cross_split_exact_duplicate"


def mmseqs_conflicts(samples, outdir, identity, coverage, threads):
    if shutil.which("mmseqs") is None:
        raise RuntimeError("MMseqs2 is not on PATH. Install/load it, or explicitly choose --similarity none")
    ids = {split: [] for split in SPLITS}
    for s in samples:
        if not s.removed:
            ids[s.split].append(s)
    if any(not ids[split] for split in SPLITS):
        raise ValueError("A split became empty; cannot audit similarity")
    pair_files = []
    with tempfile.TemporaryDirectory(prefix="fishgue_mmseqs_", dir=outdir) as temp:
        base = Path(temp)
        for split in SPLITS:
            with (base / f"{split}.fa").open("w") as fh:
                for s in ids[split]:
                    fh.write(f">{s.sid}\n{s.sequence}\n")
        for query, target in (("val", "train"), ("test", "train"), ("val", "test")):
            result = base / f"{query}_vs_{target}.tsv"
            cmd = ["mmseqs", "easy-search", str(base / f"{query}.fa"),
                   str(base / f"{target}.fa"), str(result), str(base / f"tmp_{query}_{target}"),
                   "--search-type", "3", "--min-seq-id", str(identity),
                   "--alignment-mode", "3", "-c", str(coverage), "--cov-mode", "0",
                   "-s", "7", "--mask", "0", "--strand", "2", "--threads", str(threads),
                   "--max-seqs", "100000", "--format-output", "query,target,fident,qcov,tcov"]
            subprocess.run(cmd, check=True)
            pair_files.append((result, query, target))
        conflicts = []
        for result, query, target in pair_files:
            if not result.exists():
                continue
            with result.open() as fh:
                for line in fh:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 5:
                        raise ValueError(f"Unexpected MMseqs2 output: {line[:100]}")
                    a, b = int(fields[0]), int(fields[1])
                    pident, qcov, tcov = map(float, fields[2:5])
                    if pident >= identity and min(qcov, tcov) >= coverage:
                        conflicts.append((a, b, pident, qcov, tcov, query, target))
    return conflicts


def purge_near_duplicates(samples, conflicts):
    # Train takes precedence; test takes precedence over validation. Delete
    # conflicting records, without silently reassigning their genomic blocks.
    for a, b, *_ in conflicts:
        sa, sb = samples[a], samples[b]
        if sa.removed or sb.removed:
            continue
        if sa.split == sb.split:
            continue
        loser = sa if sa.split == "val" else (sa if sa.split == "test" and sb.split == "train" else sb)
        loser.removed = "cross_split_near_duplicate"


def audit(samples, min_gap):
    by_chr = defaultdict(list)
    for s in samples:
        if not s.removed:
            by_chr[(s.species, s.chrom)].append(s)
    for namespace, rows in by_chr.items():
        rows.sort(key=lambda x: (x.start, x.end))
        max_end = {split: -10**20 for split in SPLITS}
        for s in rows:
            for other in SPLITS:
                if other != s.split and s.start - max_end[other] < min_gap:
                    raise AssertionError(f"Cross-split coordinate gap < {min_gap} in {namespace} near {s.start}")
            max_end[s.split] = max(max_end[s.split], s.end)
    exact = defaultdict(set)
    for s in samples:
        if not s.removed and s.sequence:
            exact[canonical(s.sequence)].add(s.split)
    if any(len(v) > 1 for v in exact.values()):
        raise AssertionError("Cross-split identical sequence remained")


def write_outputs(samples, schemas, task_sources, outdir, args, cost, near_hits):
    by_source = defaultdict(list)
    for s in samples:
        by_source[(s.task, s.species, s.source_bed, s.source_csv)].append(s)
    summary = []
    for (task, species, bed, source_csv), rows in sorted(by_source.items(), key=lambda x: str(x[0])):
        for split in SPLITS:
            chosen = [s for s in rows if s.split == split and not s.removed]
            summary.append({"task": task, "species": species, "source": str(bed),
                            "split": split, "retained": len(chosen),
                            "positive": sum(s.label == "1" for s in chosen),
                            "negative": sum(s.label == "0" for s in chosen),
                            "removed_boundary": sum(s.split == split and s.removed == "cross_split_boundary" for s in rows),
                            "removed_exact": sum(s.split == split and s.removed == "cross_split_exact_duplicate" for s in rows),
                            "removed_near": sum(s.split == split and s.removed == "cross_split_near_duplicate" for s in rows)})
    by_task = defaultdict(list)
    for s in samples:
        by_task[s.task].append(s)
    for task, rows in by_task.items():
        taskdir = outdir / task
        taskdir.mkdir(parents=True, exist_ok=True)
        sources = {s.source_csv for s in rows}
        if len(sources) > 1 and None in sources:
            raise ValueError(f"{task}: cannot combine BED-only and CSV sources")
        headers = {tuple(schemas[src]) for src in sources if src}
        if len(headers) > 1:
            raise ValueError(f"{task}: incompatible CSV headers across species")
        stems = {src.stem if src else s.source_bed.stem for s in rows for src in [s.source_csv]}
        basename = stems.pop() if len(stems) == 1 else task
        for split in SPLITS:
            chosen = [s for s in rows if s.split == split and not s.removed]
            with (taskdir / f"{basename}_{split}.bed").open("w") as fh:
                for s in chosen:
                    fh.write("\t".join(s.bed_fields) + "\n")
            if headers:
                with (taskdir / f"{basename}_{split}.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(next(iter(headers))))
                    writer.writeheader()
                    for s in chosen:
                        writer.writerow(s.csv_fields)
    with (outdir / "split_summary.tsv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    with (outdir / "split_manifest.tsv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["sample_id", "task", "species", "chrom", "start", "end", "source_bed", "source_csv",
                  "csv_row_index_0based", "gene_id", "locus_id", "block_index", "group", "split", "excluded_reason"]
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for s in samples:
            writer.writerow(dict(sample_id=s.sid, task=s.task, species=s.species, chrom=s.chrom,
                                 start=s.start, end=s.end, source_bed=s.source_bed, source_csv=s.source_csv or "",
                                 csv_row_index_0based=s.row_index, gene_id=s.gene, locus_id=s.locus,
                                 block_index=s.block[2], group=str(s.group), split=s.split,
                                 excluded_reason=s.removed))
    metadata = {"seed": args.seed, "target_ratios": RATIOS, "block_bp": args.block_bp,
                "boundary_buffer_bp_per_side": args.buffer_bp,
                "similarity": args.similarity, "identity": args.identity,
                "coverage_of_both_sequences": args.coverage,
                "mmseqs_detected_cross_split_pairs_before_purge": near_hits,
                "balance_score": cost, "row_alignment_user_confirmed": args.confirm_row_alignment}
    (outdir / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--task-map", required=True, type=Path)
    parser.add_argument("--chrom-species-map", type=Path, help="TSV columns chrom,species for mixed-species BED3")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--confirm-row-alignment", action="store_true",
                        help="I checked that each BED line corresponds to the same-numbered unsplit CSV row")
    parser.add_argument("--block-bp", type=int, default=100000)
    parser.add_argument("--buffer-bp", type=int, default=2500)
    parser.add_argument("--max-interval-bp", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tries", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8, help="MMseqs2 CPU threads")
    parser.add_argument("--similarity", choices=("mmseqs", "none"), default="mmseqs",
                        help="'none' checks exact duplicates only; does NOT exclude near duplicates")
    parser.add_argument("--identity", type=float, default=0.8)
    parser.add_argument("--coverage", type=float, default=0.8)
    args = parser.parse_args()
    if min(args.block_bp, args.buffer_bp, args.max_interval_bp, args.tries, args.threads) <= 0:
        parser.error("block/buffer/max interval/tries must be positive")
    if args.max_interval_bp >= args.block_bp:
        parser.error("max interval must be shorter than a block")
    if not (0 < args.identity <= 1 and 0 < args.coverage <= 1):
        parser.error("identity/coverage must be in (0,1]")
    outdir = args.output_root.resolve()
    if outdir.exists():
        parser.error(f"Output exists; choose a NEW directory: {outdir}")
    if outdir == Path(args.input_root).resolve() or Path(args.input_root).resolve() in outdir.parents:
        parser.error("Output must not be inside the original input tree")
    samples, schemas, task_sources = load_samples(args)
    if args.similarity == "mmseqs" and shutil.which("mmseqs") is None:
        parser.error("MMseqs2 not found on PATH; install/load it or explicitly use --similarity none")
    groups = make_groups(samples)
    cost = assign(groups, args.seed, args.tries)
    purge_boundaries(samples, args.buffer_bp, args.block_bp)
    purge_exact_duplicates(samples)
    outdir.mkdir(parents=True)
    near_hits = 0
    try:
        if args.similarity == "mmseqs":
            conflicts = mmseqs_conflicts(samples, outdir, args.identity, args.coverage, args.threads)
            near_hits = len(conflicts)
            purge_near_duplicates(samples, conflicts)
            # Search again against final sets. Fast heuristic search can still miss
            # matches; this only certifies detections under the chosen MMseqs run.
            remaining = mmseqs_conflicts(samples, outdir, args.identity, args.coverage, args.threads)
            if remaining:
                raise AssertionError(f"{len(remaining)} detectable near-duplicate pairs remain")
        audit(samples, 2 * args.buffer_bp)
        summary = write_outputs(samples, schemas, task_sources, outdir, args, cost, near_hits)
    except Exception:
        shutil.rmtree(outdir)
        raise
    print(f"Input: {len(samples)} rows, {len(groups)} genomic groups. Output: {outdir}")
    print(f"Retained: {sum(x['retained'] for x in summary)} rows; similarity={args.similarity}; seed={args.seed}")
    print("Review split_summary.tsv and split_manifest.tsv before training.")


if __name__ == "__main__":
    main()
