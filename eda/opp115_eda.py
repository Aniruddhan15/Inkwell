#!/usr/bin/env python3
"""
OPP-115 exploratory data analysis.

Usage:
    python opp115_eda.py --root path/to/OPP-115            # extracted folder
    python opp115_eda.py --root path/to/OPP-115_v1_0.zip   # or the zip itself
    python opp115_eda.py --root OPP-115 --threshold 0.75 --tokenizer roberta-base

Folders read:
    sanitized_policies/   policy text (segments separated by |||)
    annotations/          raw per-annotator practices  -> labels, agreement
    consolidation/        merged practices at 3 thresholds -> spans
    documentation/        websites_opp115.csv, policies_opp115.csv, errant_span_indexes/

Outputs (in --out, default eda_outputs/):
    summary.txt, tables/*.csv, figures/*.png

Notes:
  * Policies are keyed by the filename prefix (e.g. 1017_sci-news.com.csv -> "1017").
    The policy-ID column inside the annotation CSVs does NOT match the filenames
    in this release, so it is ignored.
  * Category labels are built by majority vote (>= 2 of 3 annotators) from
    annotations/, because consolidation/ only removes duplicates and keeps
    single-annotator "singlet" practices.
"""
import argparse
import html
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

ANN_COLS = ["ann_id", "batch", "annotator", "policy_col", "segment_id",
            "category", "attrs", "url", "html"]
THRESHOLDS = ["0.5", "0.75", "1.0"]
VALUE_ATTRIBUTES = ["Personal Information Type", "Third Party Entity", "Purpose",
                    "Retention Period", "Security Measure", "Choice Type"]

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class Reporter:
    def __init__(self, path):
        self.path = path
        self.lines = []

    def say(self, text=""):
        print(text)
        self.lines.append(text)

    def header(self, text):
        self.say("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)

    def save(self):
        self.path.write_text("\n".join(self.lines), encoding="utf-8")


def resolve_root(path):
    p = Path(path)
    if p.suffix.lower() == ".zip":
        out = Path(tempfile.mkdtemp(prefix="opp115_"))
        wanted = ("/annotations/", "/consolidation/", "/sanitized_policies/", "/documentation/")
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.startswith("__MACOSX"):
                    continue
                if any(w in name for w in wanted):
                    z.extract(name, out)
        found = list(out.glob("*/annotations"))
        if not found:
            sys.exit("Could not find annotations/ inside the zip.")
        return found[0].parent
    if (p / "annotations").exists():
        return p
    if (p / "OPP-115" / "annotations").exists():
        return p / "OPP-115"
    sys.exit(f"Could not find annotations/ under {p}")


def clean_html(s):
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def savefig(fig, figdir, name):
    fig.tight_layout()
    fig.savefig(figdir / name, dpi=130)
    plt.close(fig)


def bar_with_labels(ax, series, horizontal=True):
    if horizontal:
        sns.barplot(x=series.values, y=series.index, ax=ax, color="#4c72b0")
        for i, v in enumerate(series.values):
            ax.text(v, i, f" {v:,.0f}" if float(v).is_integer() else f" {v:.2f}",
                    va="center", fontsize=8)
    else:
        sns.barplot(x=series.index, y=series.values, ax=ax, color="#4c72b0")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_segments(root):
    rows = []
    for f in sorted((root / "sanitized_policies").glob("*.html")):
        pid = f.name.split("_")[0]
        text = f.read_text(encoding="utf-8", errors="replace")
        for i, seg in enumerate(text.split("|||")):
            rows.append(dict(pid=pid, seg=i, raw=seg, clean=clean_html(seg)))
    df = pd.DataFrame(rows)
    df["n_words"] = df["clean"].str.split().str.len()
    return df


def load_practices(folder):
    frames = []
    for f in sorted(Path(folder).glob("*.csv")):
        d = pd.read_csv(f, header=None, names=ANN_COLS, dtype=str, keep_default_na=False)
        d["pid"] = f.name.split("_")[0]
        d["seg"] = d["segment_id"].astype(int)

        def parse(x):
            try:
                return json.loads(x)
            except Exception:
                return None

        d["attrs_parsed"] = d["attrs"].map(parse)
        frames.append(d[["ann_id", "annotator", "pid", "seg", "category", "attrs_parsed"]])
    return pd.concat(frames, ignore_index=True)


def load_errant(root):
    errant = set()
    for f in (root / "documentation" / "errant_span_indexes").glob("*.csv"):
        e = pd.read_csv(f, header=None, dtype=str)
        for a, b in zip(e[0], e[1]):
            errant.add((a.strip(), re.sub(r"^\d+_", "", b.strip())))
    return errant


def flatten_attributes(pr, seg_raw, errant):
    """One row per attribute entry, with span validity checks against the raw segment."""
    rows = []
    for r in pr.itertuples():
        d = r.attrs_parsed
        if not isinstance(d, dict):
            continue
        for name, v in d.items():
            if not isinstance(v, dict):
                continue
            st = v.get("selectedText")
            s = v.get("startIndexInSegment")
            e = v.get("endIndexInSegment")
            has = (isinstance(st, str) and st.strip() not in ("", "null")
                   and isinstance(s, (int, float)) and isinstance(e, (int, float))
                   and not (s in (-1, 0) and e in (-1, 0)))
            exact = None
            if has:
                s, e = int(s), int(e)
                sg = seg_raw.get((r.pid, r.seg))
                exact = sg is not None and sg[s:e] == st
            rows.append(dict(
                ann_id=r.ann_id, pid=r.pid, seg=r.seg, category=r.category,
                attribute=name, value=v.get("value"), has_span=has, exact=exact,
                span_words=len(st.split()) if has else 0,
                consolidated=r.ann_id.startswith("C"),
                errant=(r.ann_id, name) in errant))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def fleiss_kappa_binary(n_pos, n_raters=3):
    """n_pos: array with, per item, how many raters said 'yes' (0..n_raters)."""
    n_pos = np.asarray(n_pos, dtype=float)
    n_neg = n_raters - n_pos
    p_i = (n_pos * (n_pos - 1) + n_neg * (n_neg - 1)) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_yes = n_pos.sum() / (len(n_pos) * n_raters)
    p_e = p_yes ** 2 + (1 - p_yes) ** 2
    if p_e == 1:
        return np.nan
    return (p_bar - p_e) / (1 - p_e)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OPP-115 folder or the zip file")
    ap.add_argument("--out", default="eda_outputs")
    ap.add_argument("--threshold", default="0.75", choices=THRESHOLDS,
                    help="consolidation threshold used for the span/value analysis")
    ap.add_argument("--tokenizer", default=None,
                    help="optional HF tokenizer name for real token counts (needs transformers)")
    args = ap.parse_args()

    root = resolve_root(args.root)
    out = Path(args.out)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    R = Reporter(out / "summary.txt")
    R.say(f"OPP-115 root: {root}")

    # ---------------- load ----------------
    seg = load_segments(root)
    seg_index = pd.MultiIndex.from_frame(seg[["pid", "seg"]])
    seg_raw = {(r.pid, r.seg): r.raw for r in seg.itertuples()}
    raw = load_practices(root / "annotations")
    errant = load_errant(root)

    R.header("0. DATASET OVERVIEW")
    R.say(f"Policies with text:       {seg.pid.nunique()}")
    R.say(f"Total segments:           {len(seg):,}")
    R.say(f"Empty segments:           {(seg.n_words == 0).sum()}")
    R.say(f"Raw practices (3 annot.): {len(raw):,}")
    ann_per_policy = raw.groupby("pid").annotator.nunique()
    R.say(f"Annotators per policy:    {ann_per_policy.value_counts().to_dict()}")
    if (ann_per_policy != 3).any():
        R.say("WARNING: some policies do not have exactly 3 annotators; kappa assumes 3.")
    missing = set(seg.pid) - set(raw.pid)
    R.say(f"Policies without annotations: {len(missing)}")

    # ---------------- 1. LABELS ----------------
    R.header("1. LABELS (majority vote from annotations/)")
    votes = (raw.groupby(["pid", "seg", "category"]).annotator.nunique()
             .unstack("category").reindex(seg_index).fillna(0).astype(int))
    cats = list(votes.sum().sort_values(ascending=False).index)
    votes = votes[cats]
    majority = votes >= 2

    raw_counts = raw.category.value_counts().reindex(cats)
    cat_table = pd.DataFrame({
        "raw_practices": raw_counts,
        "segments_any_annotator": (votes >= 1).sum(),
        "segments_majority": majority.sum(),
        "segments_all_three": (votes == 3).sum(),
    })
    cat_table["majority_pct_of_segments"] = (cat_table.segments_majority / len(seg) * 100).round(1)
    cat_table.to_csv(tabdir / "category_counts.csv")
    R.say(cat_table.to_string())

    n_cat = majority.sum(axis=1)
    R.say(f"\nSegments with no majority category: {(n_cat == 0).sum()} of {len(seg)}")
    R.say(f"Categories per segment (majority): {n_cat.value_counts().sort_index().to_dict()}")
    R.say(f"Single-annotator-only (segment, category) pairs dropped by majority vote: "
          f"{int(((votes == 1).sum()).sum()):,}")
    R.say(f"Rare categories (<50 majority segments): "
          f"{[c for c in cats if majority[c].sum() < 50]}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_with_labels(ax, cat_table.segments_majority)
    ax.set_title("Segments per category (majority vote)")
    ax.set_xlabel("segments")
    savefig(fig, figdir, "fig01_category_counts.png")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    vc = n_cat.value_counts().sort_index()
    bar_with_labels(ax, pd.Series(vc.values, index=vc.index.astype(str)), horizontal=False)
    ax.set_title("Categories per segment (majority vote)")
    ax.set_xlabel("number of categories")
    ax.set_ylabel("segments")
    savefig(fig, figdir, "fig02_categories_per_segment.png")

    X = majority.astype(int).values
    cooc = pd.DataFrame(X.T @ X, index=cats, columns=cats)
    cooc.to_csv(tabdir / "cooccurrence.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    off = cooc.where(~np.eye(len(cats), dtype=bool))
    sns.heatmap(off, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("Category co-occurrence within a segment (diagonal hidden)")
    savefig(fig, figdir, "fig03_cooccurrence.png")
    pairs = (off.stack().reset_index())
    pairs.columns = ["a", "b", "n"]
    pairs = pairs[pairs.a < pairs.b].sort_values("n", ascending=False).head(5)
    R.say("\nMost frequent category pairs:")
    R.say(pairs.to_string(index=False))

    # agreement
    kappas = {c: fleiss_kappa_binary(votes[c].values) for c in cats}
    kap = pd.Series(kappas, name="fleiss_kappa")
    kap.to_csv(tabdir / "agreement_kappa.csv")
    R.say("\nFleiss' kappa per category (binary: did the annotator use this category on the segment):")
    R.say(kap.round(3).to_string())
    R.say(f"Mean kappa: {kap.mean():.3f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_with_labels(ax, kap.sort_values(ascending=False))
    ax.set_xlim(0, 1)
    ax.set_title("Inter-annotator agreement per category (Fleiss' kappa)")
    savefig(fig, figdir, "fig04_kappa.png")

    # ---------------- 2. TEXT ----------------
    R.header("2. TEXT (sanitized_policies/)")
    segs_per_policy = seg.groupby("pid").size()
    R.say("Segments per policy: " + segs_per_policy.describe().round(1).to_dict().__str__())
    R.say("Words per segment:   " + seg.n_words.describe().round(1).to_dict().__str__())

    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            seg["n_tokens"] = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in seg.clean]
            tok_note = f"tokens ({args.tokenizer})"
        except Exception as ex:  # noqa: BLE001
            R.say(f"Tokenizer load failed ({ex}); using an approximate estimate instead.")
            seg["n_tokens"] = np.ceil(seg.n_words * 1.3)
            tok_note = "tokens (APPROX = words x 1.3)"
    else:
        seg["n_tokens"] = np.ceil(seg.n_words * 1.3)
        tok_note = "tokens (APPROX = words x 1.3; pass --tokenizer for real counts)"
    for lim in (128, 256, 512):
        R.say(f"Segments over {lim} {tok_note}: {(seg.n_tokens > lim).sum()} "
              f"({(seg.n_tokens > lim).mean() * 100:.1f}%)")
    length_stats = seg[["n_words", "n_tokens"]].describe(percentiles=[.5, .9, .95, .99]).round(1)
    length_stats.to_csv(tabdir / "length_stats.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(segs_per_policy, bins=25, color="#4c72b0")
    ax.set_title("Segments per policy")
    ax.set_xlabel("segments")
    ax.set_ylabel("policies")
    savefig(fig, figdir, "fig05_segments_per_policy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(seg.n_words, bins=40, color="#4c72b0")
    axes[0].set_title("Segment length (words)")
    axes[1].hist(seg.n_tokens, bins=40, color="#55a868")
    for lim in (256, 512):
        axes[1].axvline(lim, color="red", ls="--", lw=1)
    axes[1].set_title(f"Segment length ({tok_note})", fontsize=9)
    savefig(fig, figdir, "fig06_segment_length.png")

    # length by category
    len_by_cat = pd.DataFrame({c: seg.loc[majority[c].values, "n_words"].describe()
                               for c in cats}).T[["count", "mean", "50%", "max"]].round(1)
    len_by_cat.to_csv(tabdir / "length_by_category.csv")
    R.say("\nSegment length (words) by category:")
    R.say(len_by_cat.to_string())

    # distinctive words per category
    cv = CountVectorizer(stop_words="english", min_df=5, binary=True,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    Xw = cv.fit_transform(seg.clean)
    vocab = np.array(cv.get_feature_names_out())
    top_rows = []
    R.say("\nDistinctive words per category (log-odds vs. all other segments):")
    for c in cats:
        m = majority[c].values
        if m.sum() < 5:
            continue
        in_ = np.asarray(Xw[m].sum(0)).ravel()
        out_ = np.asarray(Xw[~m].sum(0)).ravel()
        score = np.log((in_ + 1) / (m.sum() + 2)) - np.log((out_ + 1) / ((~m).sum() + 2))
        score[in_ < 5] = -np.inf
        top = np.argsort(-score)[:12]
        words = list(vocab[top])
        top_rows.append(dict(category=c, top_words=", ".join(words)))
        R.say(f"  {c}: {', '.join(words[:8])}")
    pd.DataFrame(top_rows).to_csv(tabdir / "top_words_by_category.csv", index=False)

    # ---------------- 3. SPANS ----------------
    R.header("3. SPANS / ATTRIBUTES (consolidation/)")
    comp_rows, chosen = [], None
    for t in THRESHOLDS:
        folder = root / "consolidation" / f"threshold-{t}-overlap-similarity"
        pr = load_practices(folder)
        att = flatten_attributes(pr, seg_raw, errant)
        with_span = att[att.has_span]
        comp_rows.append(dict(
            threshold=t,
            practices=len(pr),
            merged_C_practices=int(pr.ann_id.str.startswith("C").sum()),
            singlet_practices=int((~pr.ann_id.str.startswith("C")).sum()),
            attribute_entries=len(att),
            entries_with_span=len(with_span),
            span_exact_match_pct=round(with_span.exact.mean() * 100, 1),
            span_mismatch=int((~with_span.exact.astype(bool)).sum()),
            mismatch_flagged_errant=int(((~with_span.exact.astype(bool)) & with_span.errant).sum()),
        ))
        if t == args.threshold:
            chosen = (pr, att)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(tabdir / "threshold_comparison.csv", index=False)
    R.say(comp.to_string(index=False))

    pr, att = chosen
    R.say(f"\nUsing threshold {args.threshold} for the rest of this section.")
    R.say(f"Attribute entries: {len(att):,}; with a usable span: {att.has_span.sum():,} "
          f"({att.has_span.mean() * 100:.1f}%)")

    # raw annotations for comparison
    att_raw = flatten_attributes(raw, seg_raw, errant)
    bad = att_raw[att_raw.has_span & ~att_raw.exact.astype(bool)]
    R.say(f"\nRaw annotations: {att_raw.has_span.sum():,} span entries, "
          f"{len(bad):,} do not match the segment text at their offsets "
          f"({len(bad) / max(att_raw.has_span.sum(), 1) * 100:.1f}%); "
          f"{int(bad.errant.sum())} of those are in errant_span_indexes/.")

    ok = att[att.has_span & att.exact.astype(bool)]
    span_len = (ok.groupby("attribute").span_words.describe()[["count", "mean", "50%", "max"]]
                .round(1).sort_values("count", ascending=False))
    span_len.to_csv(tabdir / "span_length_by_attribute.csv")
    R.say("\nSpan length (words) by attribute, exact-match spans only:")
    R.say(span_len.to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ok.span_words.clip(upper=100), bins=40, color="#4c72b0")
    ax.set_title("Span length in words (clipped at 100)")
    ax.set_xlabel("words")
    savefig(fig, figdir, "fig07_span_length.png")

    span_share = (att.groupby("attribute").has_span.mean() * 100).round(1).sort_values()
    R.say("\n% of entries that carry a span, by attribute (rest are optional / not-selected):")
    R.say(span_share.to_string())

    # attribute values
    value_rows = []
    present = [a for a in VALUE_ATTRIBUTES if a in set(att.attribute)]
    fig, axes = plt.subplots(len(present), 1, figsize=(8, 3 * len(present)))
    axes = np.atleast_1d(axes)
    R.say("\nMost frequent attribute values (excluding 'not-selected'):")
    for ax, a in zip(axes, present):
        v = att[att.attribute == a]
        n_unsel = int((v.value == "not-selected").sum())
        vc = v[v.value != "not-selected"].value.value_counts()
        for val, n in vc.items():
            value_rows.append(dict(attribute=a, value=val, count=int(n)))
        R.say(f"\n  {a}  (not-selected: {n_unsel})")
        R.say("    " + vc.head(8).to_string().replace("\n", "\n    "))
        top = vc.head(10)
        bar_with_labels(ax, top)
        ax.set_title(a, fontsize=10)
    pd.DataFrame(value_rows).to_csv(tabdir / "attribute_value_counts.csv", index=False)
    savefig(fig, figdir, "fig08_attribute_values.png")

    # ---------------- 4. POLICIES / METADATA ----------------
    R.header("4. POLICIES AND METADATA (documentation/)")
    meta = root / "documentation"
    w = pd.read_csv(meta / "websites_opp115.csv", dtype=str)
    w = w[w["Policy UID"].isin(set(seg.pid))].copy()
    R.say(f"Websites joined to a policy via Policy UID = filename prefix: {len(w)} of {seg.pid.nunique()}")
    sec_cols = list(w.columns[w.columns.get_loc("Sectoral Data"):])

    def top_sectors(row):
        vals = [str(x) for x in row[sec_cols].dropna() if str(x).strip()]
        return sorted({x.split(":")[0].strip() for x in vals})

    w["sectors"] = w.apply(top_sectors, axis=1)
    sec = w[["Policy UID", "Site Human-Readable Name", "sectors"]].explode("sectors").dropna()
    sec_counts = sec.sectors.value_counts()
    sec_counts.to_csv(tabdir / "sector_counts.csv", header=["policies"])
    R.say("\nPolicies per top-level DMOZ sector (a policy can have several):")
    R.say(sec_counts.head(15).to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_with_labels(ax, sec_counts.head(15))
    ax.set_title("Policies per top-level sector")
    savefig(fig, figdir, "fig09_sector_counts.png")

    share = majority.astype(int).groupby(seg.pid.values).mean()
    j = sec.merge(share, left_on="Policy UID", right_index=True)
    sec_share = j.groupby("sectors")[cats].mean()
    keep = sec_counts[sec_counts >= 5].index
    sec_share = sec_share.loc[sec_share.index.isin(keep)]
    sec_share.round(3).to_csv(tabdir / "sector_category_share.csv")
    if len(sec_share) > 1:
        import textwrap
        pct = (sec_share * 100).rename(columns=lambda c: textwrap.fill(c, 16))
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(pct) + 3))
        sns.heatmap(pct, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar=True,
                    linewidths=0.5, linecolor="white", annot_kws={"fontsize": 9},
                    cbar_kws={"label": "% of segments"})
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("Share of segments per category (%), by sector (sectors with >= 5 policies)",
                     pad=12)
        savefig(fig, figdir, "fig10_sector_category_share.png")

    p = pd.read_csv(meta / "policies_opp115.csv", dtype=str)
    p = p.iloc[:, :4]
    p.columns = ["pid", "url", "collected", "last_updated"]
    p = p[p.pid.isin(set(seg.pid))]
    yrs = pd.DataFrame({
        "collected_year": p.collected.str[:4].value_counts().sort_index(),
        "last_updated_year": p.last_updated.str[:4].value_counts().sort_index(),
    }).fillna(0).astype(int)
    yrs.to_csv(tabdir / "policy_years.csv")
    R.say("\nPolicies by collection year and last-updated year:")
    R.say(yrs.to_string())
    R.say("\nNote: the corpus was collected in 2015, so policies predate GDPR/CCPA-era wording.")

    R.header("DONE")
    R.say(f"Tables:  {tabdir}")
    R.say(f"Figures: {figdir}")
    R.save()


if __name__ == "__main__":
    main()
