#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount("/content/drive")

get_ipython().system('pip -q install beautifulsoup4 lxml')


# In[2]:


from pathlib import Path

C3PA_ROOT   = "/content/drive/MyDrive/C3PA_Dataset-main"
PROJECT     = Path("/content/drive/MyDrive/privacy_policy_ml")
INTERIM     = PROJECT / "data" / "interim"
SCHEMA_PATH = PROJECT / "data" / "label_schema.json"   # shared with the OPP-115 notebook
REPORT_DIR  = INTERIM / "c3pa_report"

MIN_SHARE       = 0.5     # keep a label if >= this share of the unit's annotators used it
MIN_WORDS       = 5       # shorter units are dropped (headings, single table cells)
CONFLICT_POLICY = "drop"  # same text in several files with different labels: "drop" or "union"
SKIP_META       = False   # True skips reading the HTML files for site domains (faster, no group_id)

INTERIM.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# In[3]:


import json

# Label order used by BOTH datasets. The OPP-115 notebook loads this file instead of redefining it.
CATEGORIES = ["First Party Collection/Use", "Third Party Sharing/Collection", "User Choice/Control",
              "Data Security", "International and Specific Audiences", "User Access, Edit and Deletion",
              "Policy Change", "Data Retention", "Do Not Track", "Other"]

if SCHEMA_PATH.exists():
    saved = json.loads(SCHEMA_PATH.read_text())
    assert saved["categories"] == CATEGORIES, \
        "label_schema.json on Drive differs from CATEGORIES. Fix one of them before continuing."
    print("label_schema.json found and matches.")
else:
    SCHEMA_PATH.write_text(json.dumps({"version": 1, "categories": CATEGORIES}, indent=2))
    print("Wrote", SCHEMA_PATH)

CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

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

assert set(C3PA_TO_OPP.values()) <= set(CATEGORIES), "Mapping target not in CATEGORIES"
COVERED    = sorted(set(C3PA_TO_OPP.values()), key=CAT_INDEX.get)
LABEL_MASK = [1 if c in COVERED else 0 for c in CATEGORIES]   # 0 = C3PA says nothing about this category
print("C3PA covers:", COVERED)
print("Masked for C3PA:", [c for c, m in zip(CATEGORIES, LABEL_MASK) if m == 0])


# In[4]:


import html
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

SETS = ["db", "ws"]


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
    s = s.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")   # literal backslash-n in raw text
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


# ---- The OPP-115 notebook must use these two functions unchanged ----
def text_key(s):
    """Key for duplicate detection: lowercase letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def norm_domain(url):
    """'https://www.Example.com/privacy' -> 'example.com'."""
    if not isinstance(url, str) or not url.startswith("http"):
        return ""
    d = urlparse(url).netloc.lower().split(":")[0]
    return d[4:] if d.startswith("www.") else d
# ---------------------------------------------------------------------


def load_annotations(root):
    frames = []
    for s in SETS:
        for f in sorted(sub(sub(root, "annotations"), s).glob("*.csv")):
            x = pd.read_csv(f, dtype=str, keep_default_na=False, encoding_errors="replace")
            x.columns = [c.strip() for c in x.columns]
            x["set"], x["file"] = s, f.stem
            x["row"] = range(len(x))          # original order inside the file
            frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    missing = {"RANumb", "Text", "Label"} - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSVs lack columns {missing}")
    df["Text"] = df.Text.map(clean)
    df["Label"] = df.Label.str.strip()
    df["RANumb"] = df.RANumb.str.strip()
    df["uid"] = df["set"] + "/" + df["file"]
    return df


# Bookkeeping: how many units survive each step, and what was dropped and why
funnel, dropped = [], []


def log_step(step, df):
    row = {"step": step, "units": len(df), "files": df.uid.nunique()}
    if "labels" in df:
        row.update(df.labels.explode().value_counts().to_dict())
    funnel.append(row)
    R.say(f"  -> {step}: {len(df):,} units in {df.uid.nunique()} files")


def log_drop(df, reason):
    if len(df):
        d = df[["uid", "Text"]].copy()
        d["Text"] = d.Text.str[:300]
        d["reason"] = reason
        dropped.append(d)


# In[5]:


R = Reporter(REPORT_DIR / "summary.txt")
root = resolve_root(C3PA_ROOT)
R.header("STEP 1  LOAD AND CLEAN")
ann = load_annotations(root)
R.say(f"Files: {ann.groupby('set').file.nunique().to_dict()} | raw rows: {len(ann):,}")

n_left = ann.Text.str.contains(r"\\n", regex=True).sum()
R.say(f"Rows still containing literal backslash-n: {n_left}")
assert n_left == 0

unknown = set(ann.Label) - set(C3PA_TO_OPP) - C3PA_DROP - {""}
assert not unknown, f"Labels not in mapping or drop list: {unknown}"

empty = ann[(ann.Text == "") | (ann.Label == "")]
log_drop(empty, "empty text or label")
ann = ann.drop(empty.index).reset_index(drop=True)
R.say(f"Dropped {len(empty):,} rows with empty Text/Label -> {len(ann):,} rows")


# In[6]:


R.header("STEP 2  ANNOTATOR VOTE")
# One vote per annotator per label. Deduplicating on (uid, Text, Label) instead would erase agreement.
a = ann.drop_duplicates(["uid", "Text", "RANumb", "Label"])
n_ra = a.groupby(["uid", "Text"]).RANumb.nunique().rename("n_ra")
R.say(f"Annotators per unit: {n_ra.value_counts().sort_index().to_dict()}")

multi_idx = n_ra[n_ra > 1].index
if len(multi_idx):
    m = a[a.set_index(["uid", "Text"]).index.isin(multi_idx)]
    ra_sets = m.groupby(["uid", "Text", "RANumb"]).Label.agg(frozenset)
    agree = ra_sets.groupby(level=[0, 1]).nunique().eq(1).mean()
    R.say(f"Multi-annotator units where all gave the same label set: {agree * 100:.1f}%")

v = (a.groupby(["uid", "Text", "Label"]).RANumb.nunique().rename("votes").reset_index()
       .join(n_ra, on=["uid", "Text"]))
v["share"] = v.votes / v.n_ra
n_ties = ((v.n_ra == 2) & (v.votes == 1)).sum()
R.say(f"Label votes that are 1-of-2 ties: {n_ties:,} (kept, since MIN_SHARE={MIN_SHARE} uses >=)")

kept = v[v.share >= MIN_SHARE]
order = ann.groupby(["uid", "Text"]).row.min().rename("row")
units = (kept.groupby(["uid", "Text"]).Label.agg(sorted).rename("c3pa_labels").reset_index()
            .join(n_ra, on=["uid", "Text"]).join(order, on=["uid", "Text"]))

lost = n_ra.index.difference(pd.MultiIndex.from_frame(units[["uid", "Text"]]))
log_drop(pd.DataFrame(list(lost), columns=["uid", "Text"]), "no label reached MIN_SHARE")
R.say(f"Units: {len(n_ra):,} before vote -> {len(units):,} after ({len(lost):,} lost)")

units["set"] = units.uid.str.split("/").str[0]
units = units.sort_values(["uid", "row"]).reset_index(drop=True)
log_step("after vote", units)


# In[7]:


R.header("STEP 3  MAP TO OPP-115 CATEGORIES")
units["labels"] = units.c3pa_labels.map(
    lambda ls: sorted({C3PA_TO_OPP[l] for l in ls if l in C3PA_TO_OPP}, key=CAT_INDEX.get))

unmapped = units[units.labels.str.len() == 0]
log_drop(unmapped, "only unmapped labels (Others / Methods / Non-discrimination)")
units = units.drop(unmapped.index).reset_index(drop=True)
R.say(f"Dropped {len(unmapped):,} units whose labels were all unmapped")

partial = units.c3pa_labels.map(lambda ls: any(l in C3PA_DROP for l in ls)).sum()
R.say(f"Units that kept some labels but lost a dropped one: {partial:,}")
log_step("after mapping", units)


# In[8]:


R.header("STEP 4  LENGTH FILTER")
units["n_words"] = units.Text.str.split().str.len()
short = units[units.n_words < MIN_WORDS]
R.say(f"Units under {MIN_WORDS} words: {len(short):,}")
R.say("Most common:\n" + short.Text.value_counts().head(15).to_string())
R.say("Categories lost:\n" + short.labels.explode().value_counts().to_string())

log_drop(short, f"fewer than {MIN_WORDS} words")
units = units.drop(short.index).reset_index(drop=True)
log_step("after length filter", units)


# In[9]:


R.header("STEP 5  DUPLICATES AND CONFLICTS ACROSS FILES")
units = units.sort_values(["uid", "row"]).reset_index(drop=True)
units["text_key"] = units.Text.map(text_key)
assert (units.text_key != "").all()

g = units.groupby("text_key")
units["n_copies"] = g.uid.transform("size")
conflict = units.assign(lk=units.labels.map(tuple)).groupby("text_key").lk.transform("nunique") > 1

R.say(f"Distinct texts with >1 copy: {(g.size() > 1).sum():,} ({(units.n_copies > 1).sum():,} units)")
R.say(f"...with conflicting label sets: {units.loc[conflict, 'text_key'].nunique():,} texts "
      f"({conflict.sum():,} units)")

if CONFLICT_POLICY == "drop":
    log_drop(units[conflict], "same text, conflicting labels across files")
    units = units[~conflict].copy()
elif CONFLICT_POLICY == "union":
    merged = units[conflict].groupby("text_key").labels.agg(
        lambda s: sorted({l for ls in s for l in ls}, key=CAT_INDEX.get))
    units["labels"] = [merged[k] if c else ls
                       for k, c, ls in zip(units.text_key, conflict, units.labels)]
else:
    raise ValueError("CONFLICT_POLICY must be 'drop' or 'union'")

dup = units.duplicated("text_key", keep="first")
log_drop(units[dup], "duplicate text (first copy kept)")
units = units[~dup].reset_index(drop=True)
R.say(f"Removed {dup.sum():,} duplicate copies")
log_step("after dedup", units)


# In[10]:


R.header("STEP 6  FILE METADATA")
from bs4 import BeautifulSoup


def doc_meta(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    title = soup.title.get_text(" ", strip=True)[:150] if soup.title else ""
    url = ""
    for tag, attrs, key in [("link", {"rel": "canonical"}, "href"),
                            ("meta", {"property": "og:url"}, "content")]:
        t = soup.find(tag, attrs=attrs)
        if t and t.get(key):
            url = t[key]
            break
    site = soup.find("meta", attrs={"property": "og:site_name"})
    return dict(title=title, site_name=site.get("content", "") if site else "",
                url=url, domain=norm_domain(url))


html_dirs = {s: sub(sub(root, "Htmls"), s) for s in SETS}
all_uids = sorted(ann.uid.unique())
rows = []
for i, uid in enumerate(all_uids):
    s, f = uid.split("/", 1)
    p = html_dirs[s] / f"{f}.html"
    meta = doc_meta(p) if (p.exists() and not SKIP_META) else {}
    rows.append({"doc_id": f"c3pa:{uid}", "uid": uid, "subset": s, **meta})
    if i % 50 == 0:
        print(f"  {i}/{len(all_uids)}")

docs = pd.DataFrame(rows)
for c in ["title", "site_name", "url", "domain"]:
    docs[c] = docs.get(c, pd.Series("", index=docs.index)).fillna("")
docs["n_units"] = docs.uid.map(units.groupby("uid").size()).fillna(0).astype(int)
# Group files by company. With no domain, a file is its own group (titles like "Privacy Policy" are too generic).
docs["group_id"] = np.where(docs.domain != "", docs.domain, docs.doc_id)

R.say(f"Files with a domain found: {(docs.domain != '').mean() * 100:.1f}%")
shared = docs[docs.domain != ""].groupby("domain").doc_id.nunique()
R.say(f"Domains with more than one file (must share a split): {(shared > 1).sum()}")
R.say(f"Files with 0 units left after preprocessing: {(docs.n_units == 0).sum()}")

units = units.join(docs.set_index("uid")[["doc_id", "group_id"]], on="uid")


# In[11]:


R.header("STEP 7  BUILD AND VALIDATE")
units = units.sort_values(["uid", "row"]).reset_index(drop=True)
units["unit_id"] = units.doc_id + "#" + units.groupby("uid").cumcount().astype(str).str.zfill(4)

records = [dict(
    source="c3pa",
    doc_id=r.doc_id,
    group_id=r.group_id,
    unit_id=r.unit_id,
    text=r.Text,
    labels=list(r.labels),           # category names from label_schema.json
    label_mask=LABEL_MASK,           # 1 = label known, 0 = ignore in the loss
    spans=[],                        # C3PA has no spans; the field keeps the format identical to OPP-115
    subset=r.set,
    n_words=int(r.n_words),
    n_annotators=int(r.n_ra),
    n_copies=int(r.n_copies),
    orig_labels=list(r.c3pa_labels),
) for r in units.itertuples()]

ids = [r["unit_id"] for r in records]
assert len(set(ids)) == len(ids), "unit_id not unique"
assert units.text_key.is_unique, "duplicate text left"
for r in records:
    assert r["text"] and "\\n" not in r["text"], r["unit_id"]
    assert r["labels"], r["unit_id"]
    assert all(l in COVERED for l in r["labels"]), r["unit_id"]
    assert len(r["label_mask"]) == len(CATEGORIES)
    assert r["n_words"] >= MIN_WORDS
R.say("All checks passed.")

R.say(f"\nFinal: {len(records):,} units from {units.doc_id.nunique()} files, "
      f"{units.group_id.nunique()} company groups")
R.say("Units per subset: " + str(units.set.value_counts().to_dict()))
R.say("Category counts:\n" + units.labels.explode().value_counts().to_string())
R.say(f"Units over ~512 tokens (>380 words, chunked later): {(units.n_words > 380).sum():,}")


# In[12]:


out = INTERIM / "c3pa.jsonl"
with out.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

docs.to_csv(REPORT_DIR / "c3pa_docs.csv", index=False)
pd.DataFrame(funnel).fillna(0).to_csv(REPORT_DIR / "funnel.csv", index=False)
if dropped:
    pd.concat(dropped, ignore_index=True).to_csv(REPORT_DIR / "dropped_units.csv", index=False)

R.header("DONE")
R.say(f"Data:    {out}")
R.say(f"Reports: {REPORT_DIR}")
R.save()


# In[13]:


chk = pd.read_json(INTERIM / "c3pa.jsonl", lines=True)
print(chk.shape)
print(pd.DataFrame(funnel).fillna(0).to_string(index=False))
print(pd.read_csv(REPORT_DIR / "dropped_units.csv").reason.value_counts())
chk.sample(5, random_state=0)[["unit_id", "group_id", "labels", "n_words", "text"]]

