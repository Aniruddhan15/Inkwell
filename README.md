# Inkwell

**By Token Economy** · DATA/MSML 641: Natural Language Processing, UMD (Fall 2026)

> We help students, everyday consumers, and small-business owners find out what a privacy policy actually says — and where it says nothing at all — by automatically identifying and classifying its clauses, instead of asking them to read the whole thing or trust an AI-generated summary.

Inkwell **identifies and classifies clauses** in a privacy policy against a fixed set of categories (data collection, sharing, retention, deletion rights, and so on). Every category assignment is tied to the exact clause that produced it — never a generated summary or paraphrase — and where a category has no matching clause anywhere in the policy, Inkwell reports that explicitly instead of guessing. There is no generative model and no LLM API anywhere in the pipeline.

**Quick links:** [Weekly reports](reports/) · [User evidence](evidence/) · 

---

## Problem and user

- **Who it's for:** Students, everyday consumers, and small-business owners — people who routinely have to agree to privacy policies but have no practical way to check what one actually commits to.
- **What they do today:** Skim or skip policies entirely, search with Ctrl+F, or ask a general-purpose chatbot for a summary.
- **Why it hurts:** Privacy policies are long, dense, and written to be skipped. People agree to terms they haven't read, can't easily tell what a policy actually covers, and struggle to act on rights (like deletion) the policy may or may not spell out. Small-business owners face the same problem when vetting a vendor's policy, and students face it when comparing apps and services aimed at them.


**Inkwell's difference:** every category label is anchored to a real clause the user can check, and gaps are reported as gaps, not filled in.

## What Inkwell does

Inkwell works in two stages.

**Stage 1 — Extraction and classification.** Inkwell segments a privacy policy and classifies every segment into one or more fixed categories (e.g., Third Party Sharing, Retention, Access/Edit/Deletion, Data Security). This single pass produces everything downstream:

- **Common FAQs, answered from the extracted clauses.** Questions people ask about almost any policy — "Can I delete my data, and how?", "Do they sell it to third parties?", "How long do they keep it?" — are answered directly from the classified clauses for the relevant categories, with the source text attached. These aren't generated answers; they're the clauses the classifier already found, organized by the question they answer.
- **Silence detection.** Inkwell checks the full category schema against what was actually found and flags any category with no matching clause anywhere in the policy, rather than assuming coverage. Silence is reported as silence, not filled in with a guess.

**Stage 2 — Open Q&A.** Once a policy is classified, the user can ask their own question, not just the common ones Stage 1 already covers. Inkwell retrieves the most relevant classified segment(s) for the question and returns the matching clause(s) and category label(s) — never a generated answer.

## How it works

```
Policy text ──► Segmenter ──► Segment classifier (multi-label, per-category)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                    ▼
        STAGE 1: Extraction & classification    STAGE 2: Open Q&A
        │                                        │
        ├─ FAQ answers: pull classified          ├─ Retrieve relevant classified
        │  segments for common questions         │   segment(s) for the user's question
        │  (deletion, sharing, retention, etc.)  │
        │                                        ├─ Attribute/span extractor highlights
        └─ Silence check: schema categories          the exact answering clause
           with zero matching segments
           ──► "not addressed" list
```

| Component | Description |
| --- | --- |
| Segment classifier | Fine-tuned **DistilBERT** or **RoBERTa-base** encoder with a multi-label sigmoid head, trained on OPP-115 categories |
| FAQ extraction | Maps common user questions (deletion, sharing, retention, security, etc.) to the categories that answer them, and surfaces the already-classified clauses for those categories |
| Silence detection | Compares the fixed category schema against categories actually assigned to any segment in the policy; unmatched categories are reported as gaps |
| Attribute/span extractor | BIO-tagging head that highlights the exact clause within a classified segment, used in both stages |
| Open Q&A (Stage 2) | TF-IDF/cosine retrieval over classified segments, reranked by predicted question category, returning the matching clause rather than a generated answer |
| Domain adaptation | Masked-language-model continued pretraining on C3PA and PolicyIE text, then supervised fine-tuning on their mapped labels |

## Data and licensing

| Dataset | Use | License / citation |
| --- | --- | --- |
| **OPP-115** | Backbone for the category schema, classifier, and attribute extractor | Research, teaching, and scholarship use only (CC BY-NC-style terms). Commercial license for annotation files available separately via CMU. Source: [usableprivacy.org/data](https://usableprivacy.org/data) |
| **C3PA** | Modern (2024-era) policy text: unsupervised adaptation for all categories, plus labels for 6 mapped categories | No stated reuse license on the repo as of this writing — needs direct confirmation with the authors before any public/commercial use. Source: [github.com/MaazBinMusa/C3PA_Dataset](https://github.com/MaazBinMusa/C3PA_Dataset) |
| **PolicyIE** | Retention and Security intents and slot annotations; adaptation text | License needs verification in dataset repo |
| **PolicyQA** | Open Q&A (Stage 2) | License needs verification in dataset repo |

**Citations**

```bibtex
@inproceedings{opp115,
  title     = {The Creation and Analysis of a Website Privacy Policy Corpus},
  author    = {Shomir Wilson and Florian Schaub and Aswarth Abhilash Dara and Frederick Liu and
               Sushain Cherivirala and Pedro Giovanni Leon and Mads Schaarup Andersen and
               Sebastian Zimmeck and Kanthashree Mysore Sathyendra and N. Cameron Russell and
               Thomas B. Norton and Eduard Hovy and Joel Reidenberg and Norman Sadeh},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics},
  year      = {2016},
  address   = {Berlin, Germany}
}

@inproceedings{c3pa,
  title     = {C3PA: An Open Dataset of Expert-Annotated and Regulation-Aware Privacy Policies
               to Enable Scalable Regulatory Compliance Audits},
  author    = {Maaz Bin Musa and Steven M. Winston and Garrison Allen and Jacob Schiller and
               Kevin Moore and Sean Quick and Johnathan Melvin and Padmini Srinivasan and
               Mihailis E. Diamantis and Rishab Nithyanand},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2024}
}
```

**Unified label schema:** OPP-115's categories are the backbone. C3PA (6 categories) and PolicyIE (2 categories) are mapped onto it through a lookup file at `config/label_mapping.json`. Every training and evaluation step, plus silence detection, reads from this file.

**Splits:** by `policy_id`, never by segment, to prevent leakage. Multi-annotator disagreement in OPP-115 is resolved by majority vote.

## Models

- **DistilBERT** and **RoBERTa-base** — fine-tuned as the segment classifier (multi-label) and as the attribute/span extractor (BIO tagging). Both are evaluated so we can report which is the better fit for clause classification vs. span extraction, rather than assuming one wins.
- No LLM API and no generative model at any stage — classification and extraction only.

## Evaluation

All numbers below are for the model that runs in the demo product. If the evaluated model and the shipped model ever differ, this README will say so.

| Model | Task | Metric | Result |
| --- | --- | --- | --- |
| TF-IDF + SVM baseline | Segment classification | Macro / micro F1 | TBD |
| Regex matcher baseline | Segment classification | Macro / micro F1 | TBD |
| DistilBERT, OPP-115 only | Segment classification | Per-category, macro / micro F1 | TBD |
| RoBERTa-base, OPP-115 only | Segment classification | Per-category, macro / micro F1 | TBD |
| Best encoder, domain-adapted | Segment classification | Per-category, macro / micro F1 | TBD |
| Span extractor | Attribute extraction | Span-level F1 (seqeval) | TBD |
| Silence detection | Gap flagging | Precision/recall on held-out policies with known gaps | TBD |
| FAQ extraction | Answer correctness | Tester-judged correctness on common questions | TBD |
| Open Q&A (Stage 2) | Retrieval + span accuracy | Correct segment retrieved / correct span returned | TBD |

**Core empirical comparison:** OPP-115-only vs. domain-adapted, each evaluated on old (OPP-115) and modern (held-out C3PA/PolicyIE) text, and DistilBERT vs. RoBERTa-base head-to-head on the same splits.

Error analysis and negative results will be documented in `docs/error_analysis.md` and in the weekly reports.

## Known limitations

- **Do Not Track** and **International/Specific Audiences** are not covered by any of our datasets. We either hand-label a small batch of modern policies or report this as an explicit gap.
- Silence detection can only flag categories missing from *our* schema and training data — it cannot detect legal non-compliance, and should not be presented as a compliance check.
- OPP-115 policies are older; performance on modern policies is measured, not assumed.
- FAQ extraction depends on clean classification into the relevant categories (e.g., Access/Edit/Deletion, Retention); ambiguous or boilerplate clauses may still need human judgment.
- Stage 2 Q&A is retrieval-based, so a question worded very differently from the policy's language may fail to retrieve the right segment even when the policy addresses it.

## Ethics, privacy, and risk

- **Data:** only datasets we are licensed to use; sources and citations documented above. C3PA's reuse terms need confirmation before anything beyond coursework use.
- **Personal data:** user-session artifacts are anonymized before being committed.
- **When the model is wrong:** Inkwell always shows the source clause and category so users can check it, and reports "not addressed" rather than inventing an answer. It is not legal advice.

## Cost to serve

No external API is used, so cost per request is local compute only.

## Team

**Token Economy**

| Name | Hat |
| --- | --- |
| Aniruddhan Narasimhan | Product |
| Hamshika Radhakrishnan | Engineering |
| Swetha Rathinavelu Saravanakumar | Data and Evaluation |
| Sreya Nagulapati | Users and Research |
| Gowtham Senthil | Operations |

Everyone writes code; each hat is accountable for keeping that part of the company healthy.

## Course context

Project for DATA/MSML 641 (Natural Language Processing), University of Maryland, Fall 2026. Company and product are fictional for course purposes.

---

