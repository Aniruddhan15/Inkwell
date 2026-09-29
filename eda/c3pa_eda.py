#!/usr/bin/env python
# coding: utf-8

# In[14]:


from google.colab import drive
drive.mount("/content/drive")

get_ipython().system('pip -q install beautifulsoup4 lxml transformers')


# In[15]:


from pathlib import Path

ROOT       = "/content/drive/MyDrive/C3PA_Dataset-main"
OUT        = Path("/content/drive/MyDrive/c3pa_eda_outputs_final")
TOKENIZER  = "bert-base-uncased"
OPP_TABLES = "/content/drive/MyDrive/eda_outputs/tables"
SKIP_HTML  = False

figdir, tabdir = OUT / "figures", OUT / "tables"
figdir.mkdir(parents=True, exist_ok=True)
tabdir.mkdir(parents=True, exist_ok=True)


# In[16]:


import html
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

sns.set_theme(style="whitegrid")
SETS = ["db", "ws"]
OPP_CATEGORIES = ["First Party Collection/Use", "Third Party Sharing/Collection", "User Choice/Control",
                  "Data Security", "International and Specific Audiences", "User Access, Edit and Deletion",
                  "Policy Change", "Data Retention", "Do Not Track", "Other"]

C3PA_TO_OPP = {
    "Categories of Personal Information Collected": "First Party Collection/Use",
    "Categories of Personal Information Shared / Disclosed": "Third Party Sharing/Collection",
    "Categories of Personal Information Sold": "Third Party Sharing/Collection",
    "Description of Right to Know PI Collected": "User Access, Edit and Deletion",
    "Description of Right to Know PI sold / shared": "User Access, Edit and Deletion",
    "Description of Right to Delete": "User Access, Edit and Deletion",
    "Description of Right to Correct Information": "User Access, Edit and Deletion",
    "Description of Right to Opt-out of sale of PI": "User Choice/Control",
    "Description of Right to Limit use of PI": "User Choice/Control",
    "Updated Privacy Policy": "Policy Change",
}
C3PA_DROP = {"Others", "Methods to exercise rights",
             "Description of Right to Non-discrimination on exercising rights"}


class Reporter:
    def __init__(self, path):
        self.path, self.lines = path, []

    def say(self, text=""):
        print(text)
        self.lines.append(str(text))

    def header(self, text):
        self.say("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)

    def save(self):
        self.path.write_text("\n".join(self.lines), encoding="utf-8")


def sub(p, name):
    for c in Path(p).iterdir():
        if c.name.lower() == name.lower():
            return c
    raise FileNotFoundError(f"Could not find '{name}' under {p}")


def resolve_root(path):
    p = Path(path)
    for cand in (p, p / "C3PA"):
        if cand.exists() and any(c.name.lower() == "annotations" for c in cand.iterdir()):
            return cand
    raise FileNotFoundError(f"Could not find annotations/ under {p}")


def clean(s):
    s = s.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")   # NEW: literal backslash-n in raw text
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def norm(s):
    return re.sub(r"\s+", " ", html.unescape(s)).strip().lower()


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s)


def savefig(fig, figdir, name):
    fig.tight_layout()
    fig.savefig(figdir / name, dpi=130)
    plt.show()
    plt.close(fig)


def bar_with_labels(ax, series, horizontal=True):
    if horizontal:
        sns.barplot(x=series.values, y=series.index, ax=ax, color="#4c72b0")
        for i, v in enumerate(series.values):
            ax.text(v, i, f" {v:,.0f}" if float(v).is_integer() else f" {v:.2f}", va="center", fontsize=8)
    else:
        sns.barplot(x=series.index, y=series.values, ax=ax, color="#4c72b0")


def load_annotations(root):
    frames = []
    for s in SETS:
        d = sub(sub(root, "annotations"), s)
        for f in sorted(d.glob("*.csv")):
            x = pd.read_csv(f, dtype=str, keep_default_na=False, encoding_errors="replace")
            x.columns = [c.strip() for c in x.columns]
            x["set"], x["file"] = s, f.stem
            frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    missing = {"RANumb", "Text", "Label"} - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSVs lack columns {missing}; found {list(df.columns)}")
    df["Text"] = df.Text.map(clean)
    df["Label"] = df.Label.str.strip()
    df["uid"] = df["set"] + "/" + df["file"]
    return df



R = Reporter(OUT / "summary.txt")


# In[17]:


root = resolve_root(ROOT)
R.say(f"C3PA root: {root}")

ann = load_annotations(root)
n_left = ann.Text.str.contains(r"\\n", regex=True).sum()
R.say(f"Units still containing literal backslash-n after clean: {n_left}")
R.header("0. DATASET OVERVIEW")
R.say(f"Annotation files: {ann.groupby('set').file.nunique().to_dict()}")
R.say(f"Raw rows: {len(ann):,} | empty Text: {(ann.Text == '').sum()} | empty Label: {(ann.Label == '').sum()}")
R.say(f"Exact duplicate rows (same file, Text, Label): {ann.duplicated(['uid', 'Text', 'Label']).sum()}")
#ann = ann[(ann.Text != "") & (ann.Label != "")].drop_duplicates(["uid", "Text", "Label"]).reset_index(drop=True)
ann = ann[(ann.Text != "") & (ann.Label != "")].drop_duplicates(["uid", "Text", "Label"]).reset_index(drop=True)

ra_per_file = ann.groupby("uid").RANumb.nunique()
R.say(f"Unique RANumb per file: {ra_per_file.value_counts().to_dict()}")

ra_per_unit = ann.groupby(["uid", "Text"]).RANumb.nunique()
R.say(f"RAs per unit: {ra_per_unit.value_counts().sort_index().to_dict()}")
R.say(f"Units labelled by >1 RA: {(ra_per_unit > 1).mean() * 100:.1f}%")

multi_idx = ra_per_unit[ra_per_unit > 1].index
if len(multi_idx):
    multi = ann[ann.set_index(["uid", "Text"]).index.isin(multi_idx)]
    ra_sets = multi.groupby(["uid", "Text", "RANumb"]).Label.agg(frozenset)
    same = ra_sets.groupby(level=[0, 1]).nunique().eq(1).mean()
    R.say(f"Multi-RA units where every RA gave the same label set: {same * 100:.1f}%")
R.say(f"Distinct labels: {ann.Label.nunique()}")

crawl = {}
for s in SETS:
    p = sub(root, "crawl") / f"{s}.csv"
    crawl[s] = pd.read_csv(p, dtype=str, encoding_errors="replace") if p.exists() else pd.DataFrame()
for s in SETS:
    n_html = len(list(sub(sub(root, "Htmls"), s).glob("*.html")))
    n_csv = ann[ann["set"] == s].file.nunique()
    R.say(f"{s}: {n_csv} annotation files with rows | {n_html} html | {len(crawl[s])} crawl rows")


MIN_SHARE = 0.5   # 0.5 = strict majority (2/3, 2/2, 1/1); 0.0 = union
votes = (ann.drop_duplicates(["uid", "Text", "RANumb", "Label"])
            .groupby(["uid", "Text", "Label"]).RANumb.nunique().rename("votes").reset_index()
            .join(ra_per_unit.rename("n_ra"), on=["uid", "Text"]))
kept = votes[votes.votes / votes.n_ra > MIN_SHARE]
units = kept.groupby(["uid", "Text"], sort=False).Label.agg(sorted).reset_index()
R.say(f"Units after vote (MIN_SHARE={MIN_SHARE}): {len(units):,} of {len(ra_per_unit):,}")
units["set"] = units.uid.str.split("/").str[0]
units["n_words"] = units.Text.str.split().str.len()
R.say(f"Text units (file, Text): {len(units):,}")


# In[18]:


R.header("1. LABELS")
ex = units.Label.explode()
Y = pd.get_dummies(ex).groupby(level=0).max().astype(int).reindex(units.index).fillna(0).astype(int)
labs = list(Y.sum().sort_values(ascending=False).index)
Y = Y[labs]

tab = pd.DataFrame({"raw_rows": ann.Label.value_counts().reindex(labs), "units": Y.sum()})
for s in SETS:
    tab[f"units_{s}"] = Y[(units["set"] == s).values].sum()
tab["pct_of_units"] = (tab.units / len(units) * 100).round(1)
tab.to_csv(tabdir / "label_counts.csv")
R.say(tab.to_string())

n_lab = Y.sum(axis=1)
R.say(f"\nLabels per unit: {n_lab.value_counts().sort_index().to_dict()}")
R.say(f"Rare labels (<50 units): {[l for l in labs if Y[l].sum() < 50]}")

fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labs))))
bar_with_labels(ax, tab.units)
ax.set_title("Units per label")
savefig(fig, figdir, "fig01_label_counts.png")

fig, ax = plt.subplots(figsize=(5, 3.5))
vc = n_lab.value_counts().sort_index()
bar_with_labels(ax, pd.Series(vc.values, index=vc.index.astype(str)), horizontal=False)
ax.set_title("Labels per unit")
savefig(fig, figdir, "fig02_labels_per_unit.png")


# In[19]:


Xm = Y.values
cooc = pd.DataFrame(Xm.T @ Xm, index=labs, columns=labs)
cooc.to_csv(tabdir / "cooccurrence.csv")
off = cooc.where(~np.eye(len(labs), dtype=bool))

fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labs)), max(7, 0.4 * len(labs))))
sns.heatmap(off, annot=len(labs) <= 20, fmt=".0f", cmap="Blues", ax=ax, cbar=False)
ax.tick_params(labelsize=6)
ax.set_title("Label co-occurrence within a unit (diagonal hidden)")
savefig(fig, figdir, "fig03_cooccurrence.png")

pairs = off.stack().reset_index()
pairs.columns = ["a", "b", "n"]
pairs = pairs[pairs.a < pairs.b].sort_values("n", ascending=False).head(8)
R.say("\nMost frequent label pairs:\n" + pairs.to_string(index=False))

#R.say("\nAgreement: C3PA has a single annotation per unit -> no kappa / majority vote possible.")
conflict = units.assign(k=units.Label.map(tuple)).groupby("Text").k.nunique()
R.say(f"Noise proxy: texts appearing in >1 file with different label sets: {(conflict > 1).sum()}")

share = tab[[f"units_{s}" for s in SETS]].div(tab[[f"units_{s}" for s in SETS]].sum())
share["abs_diff"] = (share.iloc[:, 0] - share.iloc[:, 1]).abs()
share.round(3).to_csv(tabdir / "label_share_db_vs_ws.csv")
R.say("\nLargest db-vs-ws label share differences:\n" +
      share.sort_values("abs_diff", ascending=False).head(6).round(3).to_string())

fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labs))))
share[[f"units_{s}" for s in SETS]].loc[labs[::-1]].plot.barh(ax=ax)
ax.set_title("Label share within db vs ws")
savefig(fig, figdir, "fig04_label_share_db_ws.png")


# In[20]:


R.header("2. TEXT UNITS")
upf = units.groupby("uid").size()
R.say("Units per file: " + str(upf.describe().round(1).to_dict()))
R.say("Words per unit: " + str(units.n_words.describe().round(1).to_dict()))

if TOKENIZER:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        units["n_tokens"] = [len(x) for x in tok(units.Text.tolist(), add_special_tokens=True)["input_ids"]]
        tok_note = f"tokens ({TOKENIZER})"
    except Exception as ex_:
        R.say(f"Tokenizer load failed ({ex_}); using words x 1.3")
        units["n_tokens"] = np.ceil(units.n_words * 1.3)
        tok_note = "tokens (APPROX)"
else:
    units["n_tokens"] = np.ceil(units.n_words * 1.3)
    tok_note = "tokens (APPROX = words x 1.3)"

for lim in (128, 256, 512):
    R.say(f"Units over {lim} {tok_note}: {(units.n_tokens > lim).sum()} ({(units.n_tokens > lim).mean() * 100:.1f}%)")

pct = [.5, .9, .95, .99]
length_stats = units[["n_words", "n_tokens"]].describe(percentiles=pct).round(1)
length_stats.to_csv(tabdir / "length_stats.csv")
R.say(length_stats.to_string())

short = units[units.n_words <= 6]
R.say(f"\nHeading-like units (<=6 words): {len(short)} ({len(short) / len(units) * 100:.1f}%), "
      f"all-caps: {short.Text.str.isupper().sum()}")
R.say("Most common short units:\n" + short.Text.value_counts().head(10).to_string())
R.say("(OPP-115 segments are paragraph-level; consider dropping/merging headings before combining.)")


# In[21]:


fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(upf, bins=25, color="#4c72b0")
ax.set_title("Units per file")
savefig(fig, figdir, "fig05_units_per_file.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(units.n_words.clip(upper=600), bins=40, color="#4c72b0")
axes[0].set_title("Unit length (words, clipped 600)")
axes[1].hist(units.n_tokens.clip(upper=800), bins=40, color="#55a868")
for lim in (256, 512):
    axes[1].axvline(lim, color="red", ls="--", lw=1)
axes[1].set_title(f"Unit length ({tok_note})", fontsize=9)
savefig(fig, figdir, "fig06_unit_length.png")


# In[22]:


len_by_lab = pd.DataFrame({l: units.loc[Y[l].values == 1, "n_words"].describe() for l in labs}
                          ).T[["count", "mean", "50%", "max"]].round(1)
len_by_lab.to_csv(tabdir / "length_by_label.csv")
R.say("\nUnit length (words) by label:\n" + len_by_lab.to_string())

cv = CountVectorizer(stop_words="english", min_df=5, binary=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
Xw = cv.fit_transform(units.Text)
vocab = np.array(cv.get_feature_names_out())

top_rows = []
R.say("\nDistinctive words per label (log-odds vs all other units):")
for i, l in enumerate(labs):
    m = Y[l].values == 1
    if m.sum() < 5:
        continue
    in_ = np.asarray(Xw[m].sum(0)).ravel()
    out_ = np.asarray(Xw[~m].sum(0)).ravel()
    score = np.log((in_ + 1) / (m.sum() + 2)) - np.log((out_ + 1) / ((~m).sum() + 2))
    score[in_ < 5] = -np.inf
    words = list(vocab[np.argsort(-score)[:12]])
    top_rows.append(dict(label=l, top_words=", ".join(words)))
    R.say(f"  {l}: {', '.join(words[:8])}")
pd.DataFrame(top_rows).to_csv(tabdir / "top_words_by_label.csv", index=False)


# In[23]:


def html_stats(root, ann, crawl, R):
    """Visible-text alignment of annotated units against the matching HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        R.say("bs4 not installed -> skipping HTML section")
        return pd.DataFrame()
    rows = []
    for s in SETS:
        files = sorted(sub(sub(root, "Htmls"), s).glob("*.html"),
                       key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
        for k, f in enumerate(files):
            if k % 50 == 0:
                print(f"  html {s}: {k}/{len(files)}")
            raw = f.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(raw, "lxml")
            n_tags = len(soup.find_all(True))
            title = soup.title.string.strip()[:70] if soup.title and soup.title.string else ""
            n_head = len(soup.find_all(re.compile(r"^h[1-6]$")))
            for t in soup(["script", "style", "noscript"]):
                t.decompose()
            vis = norm(soup.get_text(" "))
            vis_a = alnum(vis)
            texts = ann[(ann["set"] == s) & (ann.file == f.stem)].Text.drop_duplicates().tolist()
            exact = np.mean([norm(t) in vis for t in texts]) if texts else np.nan
            loose = np.mean([alnum(norm(t)) in vis_a for t in texts if alnum(norm(t))]) if texts else np.nan
            link, dom_ok = None, np.nan
            if f.stem.isdigit() and s in crawl and int(f.stem) - 1 < len(crawl[s]):
                link = crawl[s].Link.iloc[int(f.stem) - 1]
                m = re.search(r"https?://(?:www\.)?([^/]+)", str(link))
                if m:
                    parts = m.group(1).split(".")
                    core = parts[-2] if len(parts) >= 2 else parts[0]
                    dom_ok = core.lower() in raw.lower()
            rows.append(dict(set=s, file=f.stem, bytes=len(raw), visible_words=len(vis.split()), n_tags=n_tags,
                             n_headings=n_head, n_units=len(texts), found_exact=exact, found_loose=loose,
                             crawl_link=link, domain_in_html=dom_ok, title=title))
    return pd.DataFrame(rows)


R.header("3. SPANS AND HTML ALIGNMENT")
R.say("C3PA has no span offsets or attribute values -> cannot produce BIO extraction data.")
R.say("Substitute check: are annotated units verbatim in the matching HTML's visible text?")

hd = pd.DataFrame() if SKIP_HTML else html_stats(root, ann, crawl, R)
if len(hd):
    hd.to_csv(tabdir / "html_alignment.csv", index=False)
    R.say("\nHTML size/structure:\n" + hd.groupby("set")[["bytes", "visible_words", "n_tags", "n_headings",
          "n_units"]].describe().T.round(1).to_string())
    R.say("\nShare of annotated units found in matching HTML (exact-normalised / alnum-only):")
    R.say(hd.groupby("set")[["found_exact", "found_loose"]].describe().T.round(3).to_string())
    R.say(f"Files with <50% found (exact): {(hd.found_exact < 0.5).sum()} | (loose): {(hd.found_loose < 0.5).sum()}")
    R.say("Crawl-row -> html sanity (domain name appears in html): "
          f"{hd.domain_in_html.dropna().astype(bool).mean() * 100:.1f}% True")
    R.say("Sample of file / link / title (eyeball that they agree):\n" +
          hd.sample(min(6, len(hd)), random_state=0)[["set", "file", "crawl_link", "title"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(hd.found_loose.dropna(), bins=20, color="#4c72b0")
    ax.set_title("Share of annotated units found in HTML (alnum-only)")
    savefig(fig, figdir, "fig07_html_alignment.png")


# In[24]:


R.header("4. CRAWL METADATA (crawl/*.csv)")
for s, d in crawl.items():
    if d.empty:
        continue
    R.say(f"\n--- {s}: {d.shape} | columns {list(d.columns)}")
    R.say("Null fraction:\n" + d.isna().mean().round(3).to_string())

    dom = d.Link.str.extract(r"https?://(?:www\.)?([^/]+)")[0]
    top_tlds = dom.str.extract(r"\.([a-z]+)$")[0].value_counts().head(5).to_dict()
    https_pct = d.Link.str.startswith("https").mean() * 100
    R.say(f"Unique links {d.Link.nunique()} | unique domains {dom.nunique()} | "
          f"https {https_pct:.1f}% | top TLDs {top_tlds}")

    if "IsHomepage" in d:
        R.say(f"IsHomepage: {d.IsHomepage.value_counts(dropna=False).to_dict()}")
    for c in [c for c in d.columns if c.startswith("Textmatch")]:
        cnt = Counter(x.strip() for v in d[c].dropna() for x in v.split("||") if x.strip())
        R.say(f"{c}: {d[c].notna().sum()} non-empty; top: {dict(cnt.most_common(6))}")
    if "Link_Match" in d:
        toks = Counter(x for v in d.Link_Match.fillna("[]") for x in re.findall(r"'(\w+)'", v))
        R.say(f"Link_Match tokens: {dict(toks)}")

if all(len(crawl[s]) for s in SETS):
    R.say(f"\nLinks shared between db and ws: {len(set(crawl['db'].Link) & set(crawl['ws'].Link))}")


# In[25]:


R.header("5. READINESS FOR COMBINING WITH OPP-115")
unknown = set(labs) - set(C3PA_TO_OPP) - C3PA_DROP
assert not unknown, f"Labels missing from mapping/drop list: {unknown}"

pd.DataFrame({"c3pa_label": labs, "units": Y.sum().values,
              "opp115_category": [C3PA_TO_OPP.get(l, "DROP") for l in labs]}
             ).to_csv(tabdir / "label_mapping.csv", index=False)

units["opp_labels"] = units.Label.map(
    lambda ls: sorted({C3PA_TO_OPP[l] for l in ls if l in C3PA_TO_OPP}))
mapped = units[units.opp_labels.str.len() > 0].reset_index(drop=True)
R.say(f"Units with >=1 mapped label: {len(mapped):,} of {len(units):,} "
      f"({len(units) - len(mapped):,} dropped: only Others / Methods / Non-discrimination)")
R.say("Mapped label counts:\n" + mapped.opp_labels.explode().value_counts().to_string())
covered = set(C3PA_TO_OPP.values())
R.say(f"OPP categories C3PA cannot label (mask these in training): "
      f"{[c for c in OPP_CATEGORIES if c not in covered]}")
R.say("OPP-115 categories for reference: " + "; ".join(OPP_CATEGORIES))
R.say("Task compatibility: C3PA -> classifier only (no spans). OPP-115 -> classifier + BIO extraction.")
R.say("Split by site/file (uid), not by unit: units within a file are near-duplicates in style and content.")

a = set(units[units["set"] == "db"].Text)
b = set(units[units["set"] == "ws"].Text)
R.say(f"Identical texts appearing in both db and ws: {len(a & b)} (leakage risk if split by set)")
multi_file = units.groupby("Text").uid.nunique()
R.say(f"Texts appearing in >1 file: {(multi_file > 1).sum()} (boilerplate; consider dedup)")
R.say("Time gap: OPP-115 policies were collected in 2015 (pre-GDPR/CCPA); C3PA labels use CCPA-style "
      "wording, so expect vocabulary/label-semantics shift.")

if OPP_TABLES:
    p = Path(OPP_TABLES) / "length_stats.csv"
    if p.exists():
        o = pd.read_csv(p, index_col=0)
        cmp = pd.concat({"OPP-115 words": o["n_words"], "C3PA words": length_stats["n_words"]}, axis=1)
        cmp.to_csv(tabdir / "length_comparison_opp_vs_c3pa.csv")
        R.say("\nUnit length in words, OPP-115 segments vs C3PA units:\n" + cmp.to_string())
    else:
        R.say(f"(no length_stats.csv in {OPP_TABLES})")


# In[26]:


R.header("DONE")
R.say(f"Tables:  {tabdir}\nFigures: {figdir}")
R.save()


# In[26]:




