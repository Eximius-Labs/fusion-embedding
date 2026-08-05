"""Propagate the Tactus arXiv ID across every surface, in one pass.

Run when the ID lands (submission #7907372). Dry-run first, then live:

    PYTHONUTF8=1 uv run --env-file .env python scripts/propagate_tactus_id.py --id 2608.NNNNN
    PYTHONUTF8=1 uv run --env-file .env python scripts/propagate_tactus_id.py --id 2608.NNNNN --live

Surfaces touched automatically (each edit is anchored; the script asserts every anchor
before changing anything, and refuses to run twice):
  1. HF card EximiusLabs/fusion-embedding-2-tactus  — header link + bibtex swap to @article.
  2. HF card EximiusLabs/fusion-embedding-2-tactus-mat — paper link in ## Family.
  3. docs/tactus_outreach_sheet.md — arXiv-links block + paper-links note.
  4. papers/tactus_workshop/main.tex — "short version of" footnote on the title.
Then it prints the manual checklist (personal site pub entry, research emails 1-7 send
order, P2 optional cite, professor bundle email).

Card edits follow the card rules: patch the freshly downloaded live card only, LF
endings, show the diff before upload; uploads only with --live.
"""
import argparse
import difflib
import io
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TACTUS = "EximiusLabs/fusion-embedding-2-tactus"
MAT = "EximiusLabs/fusion-embedding-2-tactus-mat"


def patch(name: str, text: str, edits: list) -> str:
    for old, new in edits:
        if new in text:
            sys.exit(f"[{name}] already propagated (found replacement text); refusing to re-run")
        assert old in text, f"[{name}] anchor missing: {old[:80]!r}"
        assert text.count(old) == 1, f"[{name}] anchor not unique: {old[:60]!r}"
        text = text.replace(old, new, 1)
    return text


def show_diff(name: str, before: str, after: str) -> None:
    print(f"\n===== {name} =====")
    print("".join(difflib.unified_diff(before.splitlines(keepends=True),
                                       after.splitlines(keepends=True),
                                       "live", "patched", n=1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="arXiv id, e.g. 2608.12345")
    ap.add_argument("--live", action="store_true", help="apply (default: dry-run diffs only)")
    args = ap.parse_args()
    if not re.fullmatch(r"\d{4}\.\d{4,5}", args.id):
        sys.exit(f"that does not look like an arXiv id: {args.id!r}")
    aid = args.id
    url = f"https://arxiv.org/abs/{aid}"

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])

    # ---- 1. tactus card ----
    p = hf_hub_download(TACTUS, "README.md", force_download=True)
    live = io.open(p, encoding="utf-8").read()
    edits = [
        ("Tactus report: arXiv link lands with this week's submission",
         f"[Tactus report (arXiv:{aid})]({url})"),
        ("""```bibtex
@misc{tactus2026,
  title  = {Tactus: a tactile pressure sensor pack for the fusion-embedding space},
  author = {Tonmoy, Abdul Basit},
  year   = {2026},
  note   = {Eximius Labs. Model weights CC-BY-NC-4.0.},
  url    = {https://huggingface.co/EximiusLabs/fusion-embedding-2-tactus}
}
```""",
         f"""```bibtex
@article{{tactus2026,
  title   = {{Tactus: Open-Vocabulary Object Recognition from Low-Cost
             Pressure Arrays}},
  author  = {{Tonmoy, Abdul Basit}},
  journal = {{arXiv preprint arXiv:{aid}}},
  year    = {{2026}}
}}
```"""),
    ]
    patched = patch("tactus card", live, edits)
    show_diff("tactus card", live, patched)
    if args.live:
        tmp = os.path.join(os.path.dirname(p), "README_arxiv.md")
        io.open(tmp, "w", encoding="utf-8", newline="\n").write(patched)
        api.upload_file(path_or_fileobj=tmp, path_in_repo="README.md", repo_id=TACTUS,
                        commit_message=f"Link the Tactus paper (arXiv:{aid})")
        print("[tactus card] uploaded")

    # ---- 2. tactus-mat card ----
    p = hf_hub_download(MAT, "README.md", force_download=True)
    live = io.open(p, encoding="utf-8").read()
    edits = [
        ("## Family",
         f"## Paper\n\nThe Tactus recipe both profiles build on is described in "
         f"[arXiv:{aid}]({url}).\n\n## Family"),
    ]
    patched = patch("mat card", live, edits)
    show_diff("mat card", live, patched)
    if args.live:
        tmp = os.path.join(os.path.dirname(p), "README_arxiv.md")
        io.open(tmp, "w", encoding="utf-8", newline="\n").write(patched)
        api.upload_file(path_or_fileobj=tmp, path_in_repo="README.md", repo_id=MAT,
                        commit_message=f"Link the Tactus paper (arXiv:{aid})")
        print("[mat card] uploaded")

    # ---- 3. outreach sheet ----
    sheet = os.path.join(REPO_ROOT, "docs", "tactus_outreach_sheet.md")
    live = io.open(sheet, encoding="utf-8").read()
    edits = [
        ("The adapters paper and the Tactus paper are not submitted yet.",
         f"Tactus paper **arXiv:{aid}** ({url}). The adapters paper (P2) submits next."),
        ("hold those emails until it has an ID, or send now with the weights and code\nlinks they already carry.",
         f"it is live at arXiv:{aid}; drafts 1-7 are clear to send with that link added."),
        ("arXiv:2607.18666 (live); adapter paper (P2), speech-axis paper (P3), and the Tactus\nworkshop 2-pager go to arXiv this week; insert their IDs here when assigned.",
         f"arXiv:2607.18666 (live); corpus paper arXiv:2608.01560 (live); Tactus paper\narXiv:{aid} (live). Adapter paper (P2) submits next; insert its ID here when assigned."),
    ]
    patched = patch("outreach sheet", live, edits)
    show_diff("outreach sheet", live, patched)
    if args.live:
        io.open(sheet, "w", encoding="utf-8", newline="\n").write(patched)
        print("[outreach sheet] written")

    # ---- 4. workshop tex ----
    tex = os.path.join(REPO_ROOT, "papers", "tactus_workshop", "main.tex")
    live = io.open(tex, encoding="utf-8").read()
    anchor = "\\maketitle"
    assert anchor in live and live.count(anchor) == 1, "[workshop tex] maketitle anchor"
    note = (f"\\maketitle\n\\begingroup\\renewcommand\\thefootnote{{}}\\footnotetext{{"
            f"This is a short version of arXiv:{aid}.}}\\endgroup")
    if f"short version of arXiv:{aid}" in live:
        sys.exit("[workshop tex] already propagated")
    patched = live.replace(anchor, note, 1)
    show_diff("workshop tex", live, patched)
    if args.live:
        io.open(tex, "w", encoding="utf-8", newline="\n").write(patched)
        print("[workshop tex] written (recompile before submitting)")

    print(f"""
===== MANUAL CHECKLIST (after --live) =====
1. Personal site: add the Tactus pub entry (arXiv:{aid}) + update the brewing bullet.
2. Send research emails 1-7 from docs/tactus_outreach_sheet.md (Ostadabbas medical
   track is independent and may already be sent). Add the arXiv link line to each.
3. P2: optionally cite the Tactus paper before submitting; then submit P2 (tarball
   ready at papers/arxiv_submissions/p2_adapters.tar.gz, 4 authors, form metadata
   needs all four names).
4. Professor bundle email (grants): the ID is the hook; drafts per funding_research.md.
5. Workshop tex: recompile and check the footnote renders on page 1.
6. Group C outreach gate ALSO needs Fusion Perception v0.2 shipped; check both before
   sending Seeam (fix the two flagged claims first; see outreach_emails.md C1 status).
""")


if __name__ == "__main__":
    main()
