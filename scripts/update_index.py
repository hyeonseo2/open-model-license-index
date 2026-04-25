#!/usr/bin/env python3
"""
Open Model License Index — update_index.py

Collects license metadata from tracked organizations and
generates:
  - data/models.json   (full JSON with evidence)
  - data/models.csv    (flat CSV)
  - public/models.json (copy for GitHub Pages)
  - README.md          (marker-region table update)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, ModelInfo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "sources.yml"
DATA_DIR = ROOT / "data"
PUBLIC_DIR = ROOT / "public"
README_PATH = ROOT / "README.md"

MARKER_START = "<!-- MODEL_LICENSE_TABLE:START -->"
MARKER_END = "<!-- MODEL_LICENSE_TABLE:END -->"

LICENSE_FILE_PATTERNS = re.compile(
    r"^(LICENSE|LICENCE|COPYING)([-_.].*)?$", re.IGNORECASE
)

HF_BASE = "https://huggingface.co"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# License extraction helpers (NO classification, NO risk, NO interpretation)
# ---------------------------------------------------------------------------

def _find_license_files(siblings: list | None) -> list[str]:
    """Return list of repo-relative paths matching LICENSE/LICENCE/COPYING."""
    if not siblings:
        return []
    results = []
    for sib in siblings:
        name = sib.rfilename if hasattr(sib, "rfilename") else str(sib)
        if LICENSE_FILE_PATTERNS.match(name.split("/")[-1]):
            results.append(name)
    return sorted(results)


def _extract_license_from_tags(tags: list[str] | None) -> str | None:
    """Try to get a license id from hub tags like 'license:mit'."""
    if not tags:
        return None
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _get_card_data_field(card_data: Any, field: str) -> Any:
    """Safely read a field from model card data."""
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get(field)
    return getattr(card_data, field, None)


def extract_license_info(model: ModelInfo) -> dict[str, Any]:
    """
    Build the license sub-object for a model.

    Priority:
      1. model card metadata 'license' field
      2. hub tag 'license:*'
      3. 'unknown'

    No classification, no risk, no commercial-use judgment.
    """
    card_data = model.card_data
    card_license = _get_card_data_field(card_data, "license")
    card_license_name = _get_card_data_field(card_data, "license_name")
    card_license_link = _get_card_data_field(card_data, "license_link")

    # Determine raw license value and source
    if card_license:
        raw = card_license
        metadata_source = "model_card_metadata"
    else:
        tag_license = _extract_license_from_tags(model.tags)
        if tag_license:
            raw = tag_license
            metadata_source = "hub_tag"
        else:
            raw = "unknown"
            metadata_source = "none"

    # License files in repo
    license_files = _find_license_files(model.siblings)
    primary_license_file = license_files[0] if license_files else None

    model_id = model.id or ""
    license_file_url = (
        f"{HF_BASE}/{model_id}/blob/main/{primary_license_file}"
        if primary_license_file
        else None
    )

    # Build evidence list
    evidence: list[dict[str, Any]] = []

    # Evidence: model card (always present)
    evidence.append(
        {
            "kind": "model_card",
            "label": "model card",
            "url": f"{HF_BASE}/{model_id}",
        }
    )

    # Evidence: license_link if present
    if card_license_link:
        evidence.append(
            {
                "kind": "license_link",
                "label": "license link",
                "url": card_license_link,
            }
        )

    # Evidence: license files
    for lf in license_files:
        evidence.append(
            {
                "kind": "license_file",
                "label": lf,
                "path": lf,
                "url": f"{HF_BASE}/{model_id}/blob/main/{lf}",
            }
        )

    # display_url only when explicit license_link exists
    display_url = card_license_link if card_license_link else None

    return {
        "raw": raw,
        "id": raw,
        "key": raw,
        "name": card_license_name,
        "display_url": display_url,
        "metadata_source": metadata_source,
        "license_link": card_license_link,
        "license_file": primary_license_file,
        "license_file_url": license_file_url,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Model scanning
# ---------------------------------------------------------------------------

def scan_models(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    api = HfApi()
    organizations = cfg.get("organizations", [])
    collection_orgs = cfg.get("collection_organizations", [])
    per_org_limit = cfg.get("per_org_limit", 100)
    max_scan_per_org = cfg.get("max_scan_per_org", 100)
    pipeline_tags = cfg.get("pipeline_tags", [])
    include_gated = cfg.get("include_gated", True)

    include_patterns = [
        re.compile(p) for p in cfg.get("include_name_regex", []) if p
    ]
    exclude_patterns = [
        re.compile(p) for p in cfg.get("exclude_name_regex", []) if p
    ]

    now = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    print(f"[config] {len(organizations)} standard orgs, {len(collection_orgs)} collection orgs, limit={per_org_limit}")

    for org in organizations:
        print(f"[scan] org={org}")
        scanned = 0
        collected = 0

        try:
            models_iter = api.list_models(
                author=org,
                sort="last_modified",
                limit=max_scan_per_org,
                expand=["cardData", "siblings"],
            )

            for model in models_iter:
                scanned += 1
                if collected >= per_org_limit:
                    break

                model_id = model.id or ""

                # Pipeline tag filter
                if pipeline_tags:
                    pt = model.pipeline_tag or ""
                    if pt not in pipeline_tags:
                        continue

                # Regex filters
                if include_patterns:
                    if not any(p.search(model_id) for p in include_patterns):
                        continue
                if exclude_patterns:
                    if any(p.search(model_id) for p in exclude_patterns):
                        continue

                # Gated filter
                is_gated = bool(model.gated) if model.gated else False
                if not include_gated and is_gated:
                    continue

                license_info = extract_license_info(model)

                entry = {
                    "id": model_id,
                    "url": f"{HF_BASE}/{model_id}",
                    "author": model.author,
                    "tracked_org": org,
                    "likes": model.likes or 0,
                    "sha": model.sha or "",
                    "license": license_info,
                    "sources": {
                        "model_card": f"{HF_BASE}/{model_id}",
                    },
                    "last_checked_at": now,
                }
                results.append(entry)
                collected += 1

        except Exception as exc:
            print(f"[error] org={org}: {exc}", file=sys.stderr)

        print(f"  scanned={scanned}  collected={collected}")

    for org in collection_orgs:
        print(f"[scan] org={org} (collection-based)")
        scanned = 0
        collected = 0

        try:
            cols = api.list_collections(owner=org, limit=max_scan_per_org)
            for collection in cols:
                scanned += 1
                if collected >= per_org_limit:
                    break
                
                try:
                    col_details = api.get_collection(collection.slug)
                    print(f"    [col] {collection.slug}: {len(col_details.items)} items")
                except Exception as e:
                    print(f"    [error] fetch col={collection.slug}: {e}")
                    continue

                item_ids = [i.item_id for i in col_details.items if i.item_type == "model"]
                if not item_ids:
                    continue

                best_m = None
                best_dt = None

                import time
                for item_id in item_ids[:5]:
                    try:
                        time.sleep(1.0)
                        # Cheap call without expand to get last_modified (which disappears when expanded)
                        m_info = api.model_info(repo_id=item_id)
                        print(f"      [check] {item_id} (modified: {getattr(m_info, 'last_modified', 'N/A')})")
                    except Exception as exc:
                        if "429" in str(exc):
                            print(f"      [rate limit] waiting 30s...")
                            time.sleep(30)
                        else:
                            print(f"      [error] checking {item_id}: {exc}")
                        continue

                    dt_str = getattr(m_info, "last_modified", None)
                    if not dt_str:
                        continue
                    
                    if best_dt is None or dt_str > best_dt:
                        best_dt = dt_str
                        best_m = m_info

                if best_m:
                    print(f"    [winner] {best_m.id} for col {collection.slug}")
                    
                    # Full fetch only for the winner
                    try:
                        time.sleep(1.0)
                        full_m = api.model_info(repo_id=best_m.id, expand=["cardData", "siblings"])
                    except Exception as e:
                        print(f"    [error] full fetch {best_m.id}: {e}")
                        continue

                    license_info = extract_license_info(full_m)
                    model_id = full_m.id or ""
                    entry = {
                        "id": model_id,
                        "url": f"{HF_BASE}/{model_id}",
                        "author": getattr(full_m, "author", org),
                        "tracked_org": org,
                        "likes": getattr(full_m, "likes", 0) or 0,
                        "sha": getattr(full_m, "sha", "") or "",
                        "license": license_info,
                        "sources": {
                            "model_card": f"{HF_BASE}/{model_id}",
                        },
                        "last_checked_at": now,
                    }
                    results.append(entry)
                    collected += 1
        except Exception as exc:
            print(f"[error] org={org}: {exc}", file=sys.stderr)

        print(f"  scanned_cols={scanned}  collected={collected}")

    # Sort by likes descending to show top models across orgs
    results.sort(key=lambda m: m.get("likes", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Output: JSON
# ---------------------------------------------------------------------------

def write_json(models: list[dict[str, Any]]) -> None:
    data_path = DATA_DIR / "models.json"
    public_path = PUBLIC_DIR / "models.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(models),
        "models": models,
    }

    for path in (data_path, public_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[json] wrote {len(models)} models")


# ---------------------------------------------------------------------------

def _format_license_cell(lic: dict[str, Any]) -> str:
    """
    Build the License column cell with the best available link.
    """
    raw = lic.get("raw", "unknown")
    name = lic.get("name")

    if name and name != raw:
        label = f"{name} (`{raw}`)"
    else:
        label = f"`{raw}`"

    best_url = None
    evidence_list = lic.get("evidence", [])
    for kind in ("license_link", "license_file", "model_card"):
        for ev in evidence_list:
            if ev["kind"] == kind:
                best_url = ev["url"]
                break
        if best_url:
            break

    if best_url:
        label = f"[{label}]({best_url})"

    return label


# ---------------------------------------------------------------------------
# Output: README
# ---------------------------------------------------------------------------

def update_readme(models: list[dict[str, Any]], max_rows: int = 200) -> None:
    if not README_PATH.exists():
        print("[readme] README.md not found — creating from template")
        _create_readme_template()

    content = README_PATH.read_text(encoding="utf-8")

    if MARKER_START not in content or MARKER_END not in content:
        print("[readme] markers not found — recreating README.md")
        _create_readme_template()
        content = README_PATH.read_text(encoding="utf-8")

    table = _build_markdown_table(models[:max_rows])
    summary = f"\n> {len(models)} models tracked · updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    replacement = f"{MARKER_START}\n{summary}{table}\n{MARKER_END}"

    pattern = re.compile(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        re.DOTALL,
    )
    new_content = pattern.sub(replacement, content)
    README_PATH.write_text(new_content, encoding="utf-8")
    print(f"[readme] updated with {min(len(models), max_rows)} rows")


def _build_markdown_table(models: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | License |",
        "|---|---|",
    ]
    for m in models:
        lic = m.get("license", {})
        model_link = f"[`{m['id']}`]({m['url']})"
        license_cell = _format_license_cell(lic)

        lines.append(
            f"| {model_link} | {license_cell} |"
        )
    return "\n".join(lines)


def _create_readme_template() -> None:
    template = f"""# Open Model License Index

[![Update Index](https://github.com/YOUR_USER/open-model-license-index/actions/workflows/update.yml/badge.svg)](https://github.com/YOUR_USER/open-model-license-index/actions/workflows/update.yml)

> **Important:** this project does **not** classify, score, or interpret licenses.
> metadata, tags, and repository license files. Always review the linked original
> source before using a model.

## What is this?

A daily-updated index of license metadata for popular open models.

- **Data sources**: Hugging Face Hub API — model card metadata, hub tags, repository files
- **Update frequency**: Daily via GitHub Actions
- **GitHub Pages**: Browse the interactive table at your Pages URL

## Table

{MARKER_START}
<!-- auto-generated — do not edit -->
{MARKER_END}

## Files

| File | Description |
|---|---|
| `data/models.json` | Full JSON with license evidence |
| `public/models.json` | JSON served by GitHub Pages |
| `public/index.html` | Interactive browse UI |

## Configuration

Edit `config/sources.yml` to add or remove tracked organizations.

## License

This project is licensed under the [MIT License](LICENSE).
"""
    README_PATH.write_text(template, encoding="utf-8")


# ---------------------------------------------------------------------------
# Output: GitHub Pages HTML
# ---------------------------------------------------------------------------

def write_pages_html() -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    html_path = PUBLIC_DIR / "index.html"
    html_path.write_text(_PAGES_HTML, encoding="utf-8")
    print("[html] wrote public/index.html")


_PAGES_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open Model License Index</title>
<meta name="description" content="Daily-updated index of license metadata for popular open models. Browse, filter, and search model license information.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#030712; --bg-glass:rgba(17,24,39,0.7);
  --surface:rgba(30,41,59,0.5); --surface-hover:rgba(51,65,85,0.7);
  --border:rgba(255,255,255,0.08); --border-glow:rgba(99,102,241,0.3);
  --text:#f8fafc; --text2:#94a3b8;
  --accent:#818cf8; --accent2:#c084fc;
  --green:#34d399; --red:#f87171;
  --radius:16px; --radius-sm:10px;
  --shadow:0 8px 32px rgba(0,0,0,0.4);
}
html{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;background-attachment:fixed;}
body{
  min-height:100vh;padding:0;
  background-image:
    radial-gradient(circle at 15% 50%, rgba(99,102,241,0.15), transparent 25%),
    radial-gradient(circle at 85% 30%, rgba(192,132,252,0.15), transparent 25%);
}

/* Header */
.hero{
  padding:4rem 2rem 3rem;text-align:center;position:relative;overflow:hidden;
  border-bottom:1px solid var(--border);
  background:linear-gradient(to bottom, rgba(15,23,42,0.8), rgba(3,7,18,0));
}
.hero *{position:relative;z-index:1}
.hero h1{
  font-family:'Outfit',sans-serif;font-size:3.5rem;font-weight:800;margin-bottom:.75rem;
  letter-spacing:-.03em;
  background:-webkit-linear-gradient(0deg, #818cf8, #e879f9);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 4px 12px rgba(139,92,246,0.3));
}
.hero p{color:var(--text2);font-size:1.05rem;max-width:640px;margin:0 auto;line-height:1.6;}
.hero .badge{
  display:inline-block;margin-top:1.5rem;padding:.4rem 1.2rem;
  border-radius:99px;background:rgba(255,255,255,.05);border:1px solid var(--border);
  backdrop-filter:blur(12px);font-size:.85rem;color:var(--text2);font-weight:500;
  box-shadow:0 2px 10px rgba(0,0,0,0.2);
}

/* Container */
.container{max-width:1200px;margin:0 auto;padding:2rem}

/* Controls bar */
.controls{
  display:flex;flex-wrap:wrap;gap:1rem;margin-bottom:1.5rem;align-items:center;
}
.controls input[type=text],.controls select{
  padding:.7rem 1.2rem;border-radius:var(--radius-sm);border:1px solid var(--border);
  background:var(--bg-glass);backdrop-filter:blur(8px);
  color:var(--text);font-size:.9rem;transition:all .3s ease;outline:none;
}
.controls input[type=text]:focus,.controls select:focus{
  border-color:var(--accent);box-shadow:0 0 0 3px var(--border-glow);
}
.controls input[type=text]{flex:1;min-width:240px}
.controls select{min-width:150px;cursor:pointer;}
.controls select:hover{background:var(--surface);}

/* Stats bar */
.stats{display:flex;gap:1.5rem;margin-bottom:2rem;flex-wrap:wrap;}
.stat-card{
  flex:1;min-width:160px;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.4rem;display:flex;flex-direction:column;gap:.4rem;
  backdrop-filter:blur(10px);transition:transform .3s ease,border-color .3s;box-shadow:var(--shadow);
}
.stat-card:hover{transform:translateY(-4px);border-color:var(--border-glow);}
.stat-card .label{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text2);font-weight:600;}
.stat-card .value{font-size:1.8rem;font-weight:700;font-family:'Outfit',sans-serif;color:var(--text);}

/* Table */
.table-wrap{
  overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--bg-glass);backdrop-filter:blur(16px);box-shadow:var(--shadow);
}
table{width:100%;border-collapse:separate;border-spacing:0;white-space:nowrap}
thead{position:sticky;top:0;z-index:2;}
th{
  background:rgba(15,23,42,0.95);backdrop-filter:blur(8px);
  padding:1.2rem 1.4rem;text-align:left;font-weight:600;font-size:.8rem;
  text-transform:uppercase;letter-spacing:.05em;color:var(--text2);
  cursor:pointer;user-select:none;border-bottom:1px solid var(--border);
  transition:color .2s;
}
th:hover{color:var(--accent)}
th .sort-icon{margin-left:.4rem;opacity:.4;font-size:.7rem}
td{
  padding:1rem 1.4rem;border-bottom:1px solid rgba(255,255,255,0.04);font-size:.85rem;color:var(--text);
  transition:background .2s;
}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--surface-hover);}

/* Links & Code */
td a{color:var(--accent2);text-decoration:none;font-weight:500;transition:all .2s;}
td a:hover{color:#d8b4fe;text-shadow:0 0 8px rgba(216,180,254,0.4);}
td code{
  background:rgba(255,255,255,0.06);padding:.25rem .6rem;border-radius:6px;
  font-family:ui-monospace,monospace;font-size:.8rem;color:var(--green);
  border:1px solid rgba(255,255,255,0.05);
}

.no-results{text-align:center;padding:4rem;color:var(--text2);font-size:1rem;font-weight:500;}

/* Footer */
footer{text-align:center;padding:3rem 2rem;color:var(--text2);font-size:.85rem;border-top:1px solid var(--border);margin-top:4rem;}
footer a{color:var(--accent);text-decoration:none;transition:color .2s;}
footer a:hover{color:var(--text);}

/* Responsive */
@media(max-width:768px){
  .hero{padding:3rem 1rem 2rem}
  .hero h1{font-size:2.2rem}
  .container{padding:1rem}
  .controls{flex-direction:column}
  .controls input[type=text],.controls select{width:100%}
  .stats{gap:.75rem}
}
</style>
</head>
<body>

<div class="hero">
  <h1>🌐 Open Model License Index</h1>
  <p>Daily-updated license metadata for popular open models.<br>
  This project does <strong>not</strong> classify or interpret licenses.</p>
  <span class="badge" id="updateBadge">Loading…</span>
</div>

<div class="container">
  <div class="controls">
    <input type="text" id="searchInput" placeholder="Search models, orgs, licenses…" autofocus>
    <select id="orgFilter"><option value="">All Orgs</option></select>
    <select id="sourceFilter"><option value="">All Sources</option></select>
    <select id="licFileFilter">
      <option value="">License file: Any</option>
      <option value="yes">Has license file</option>
      <option value="no">No license file</option>
    </select>
  </div>

  <div class="stats">
    <div class="stat-card"><span class="label">Total Models</span><span class="value" id="statTotal">—</span></div>
    <div class="stat-card"><span class="label">Shown</span><span class="value" id="statShown">—</span></div>
    <div class="stat-card"><span class="label">Organizations</span><span class="value" id="statOrgs">—</span></div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th data-col="id">Model <span class="sort-icon">⇅</span></th>
          <th data-col="org">Org <span class="sort-icon">⇅</span></th>
          <th data-col="license">License <span class="sort-icon">⇅</span></th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="no-results" id="noResults" style="display:none">No models match your filters.</div>
  </div>
</div>

<footer>
  <p>Data sourced from <a href="https://huggingface.co" target="_blank">Hugging Face Hub</a>.
  This project does <strong>not</strong> classify, score, or interpret licenses.<br>
  <a href="https://github.com/YOUR_USER/open-model-license-index" target="_blank">GitHub Repository</a></p>
</footer>

<script>
(function(){
"use strict";
let allModels=[], sortCol="id", sortDir=1;

fetch("models.json")
  .then(r=>r.json())
  .then(data=>{
    allModels=data.models||[];
    document.getElementById("updateBadge").textContent=
      `${data.count} models · updated ${(data.generated_at||'').slice(0,16).replace('T',' ')} UTC`;
    populateFilters();
    renderTable();
  })
  .catch(()=>{
    document.getElementById("noResults").style.display="block";
    document.getElementById("noResults").textContent="Failed to load models.json";
  });

function populateFilters(){
  const orgs=new Set(), sources=new Set();
  allModels.forEach(m=>{
    if(m.tracked_org) orgs.add(m.tracked_org);
    if(m.license&&m.license.metadata_source) sources.add(m.license.metadata_source);
  });
  const orgSel=document.getElementById("orgFilter");
  [...orgs].sort().forEach(o=>{const opt=document.createElement("option");opt.value=o;opt.textContent=o;orgSel.appendChild(opt)});
  const srcSel=document.getElementById("sourceFilter");
  [...sources].sort().forEach(s=>{const opt=document.createElement("option");opt.value=s;opt.textContent=s;srcSel.appendChild(opt)});
}

function getFiltered(){
  const q=document.getElementById("searchInput").value.toLowerCase();
  const org=document.getElementById("orgFilter").value;
  const src=document.getElementById("sourceFilter").value;
  const lf=document.getElementById("licFileFilter").value;
  return allModels.filter(m=>{
    if(org && m.tracked_org!==org) return false;
    const lic=m.license||{};
    if(src && lic.metadata_source!==src) return false;
    if(lf==="yes" && !lic.license_file) return false;
    if(lf==="no" && lic.license_file) return false;
    if(q){
      const hay=[m.id,m.tracked_org,lic.raw,lic.name||"",lic.metadata_source||""].join(" ").toLowerCase();
      if(!hay.includes(q)) return false;
    }
    return true;
  });
}

function sortModels(list){
  return list.sort((a,b)=>{
    let va,vb;
    switch(sortCol){
      case"id":va=a.id;vb=b.id;break;
      case"org":va=a.tracked_org;vb=b.tracked_org;break;
      case"license":va=(a.license||{}).raw||"";vb=(b.license||{}).raw||"";break;
      default:va=a.id;vb=b.id;
    }
    if(typeof va==="string") return sortDir*va.localeCompare(vb);
    return sortDir*((va>vb?1:va<vb?-1:0));
  });
}

function buildLicenseCell(lic){
  const raw=lic.raw||"unknown";
  const name=lic.name;
  let label=name&&name!==raw?`${escHtml(name)} (<code>${escHtml(raw)}</code>)`:`<code>${escHtml(raw)}</code>`;
  
  let bestUrl = null;
  const ev = lic.evidence || [];
  for(const kind of ["license_link", "license_file", "model_card"]){
    const found = ev.find(e=>e.kind===kind);
    if(found){
      bestUrl = found.url;
      break;
    }
  }
  
  if(bestUrl) label=`<a href="${escAttr(bestUrl)}" target="_blank">${label}</a>`;
  return label;
}

function escHtml(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function escAttr(s){return s.replace(/&/g,"&amp;").replace(/"/g,"&quot;")}

function renderTable(){
  const filtered=sortModels(getFiltered());
  const tbody=document.getElementById("tbody");
  const noRes=document.getElementById("noResults");

  document.getElementById("statTotal").textContent=allModels.length;
  document.getElementById("statShown").textContent=filtered.length;
  document.getElementById("statOrgs").textContent=new Set(allModels.map(m=>m.tracked_org)).size;

  if(!filtered.length){tbody.innerHTML="";noRes.style.display="block";return}
  noRes.style.display="none";

  const rows=filtered.map(m=>{
    const lic=m.license||{};
    return `<tr>
      <td><a href="${escAttr(m.url)}" target="_blank"><code>${escHtml(m.id)}</code></a></td>
      <td>${escHtml(m.tracked_org||"")}</td>
      <td>${buildLicenseCell(lic)}</td>
    </tr>`;
  });
  tbody.innerHTML=rows.join("");
}

// Event listeners
["searchInput","orgFilter","sourceFilter","licFileFilter"].forEach(id=>{
  document.getElementById(id).addEventListener("input",renderTable);
  document.getElementById(id).addEventListener("change",renderTable);
});

document.querySelectorAll("th[data-col]").forEach(th=>{
  th.addEventListener("click",()=>{
    const col=th.dataset.col;
    if(sortCol===col) sortDir*=-1; else{sortCol=col;sortDir=1}
    renderTable();
  });
});

})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Output: GitHub Actions workflow
# ---------------------------------------------------------------------------

def write_github_actions() -> None:
    wf_dir = ROOT / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    wf_path = wf_dir / "update.yml"
    wf_path.write_text(
        """\
name: Update Index

on:
  workflow_dispatch:
  schedule:
    - cron: "20 17 * * *"

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - run: pip install -r requirements.txt

      - name: Run update script
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python scripts/update_index.py

      - name: Commit changes
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add README.md data/ public/
          git diff --cached --quiet || git commit -m "chore: update model license index [$(date -u +%Y-%m-%dT%H:%M)]"
          git push

  deploy-pages:
    needs: update
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main

      - name: Pull latest
        run: git pull origin main

      - uses: actions/configure-pages@v5

      - uses: actions/upload-pages-artifact@v3
        with:
          path: public

      - id: deployment
        uses: actions/deploy-pages@v4
""",
        encoding="utf-8",
    )
    print("[actions] wrote .github/workflows/update.yml")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Open Model License Index — update_index.py")
    print("=" * 60)

    cfg = load_config()
    orgs = cfg.get("organizations") or []
    coll_orgs = cfg.get("collection_organizations") or []
    limit = cfg.get("per_org_limit")
    print(f"[config] {len(orgs)} standard orgs, {len(coll_orgs)} collection orgs, limit={limit}")

    models = scan_models(cfg)
    print(f"[scan] total models collected: {len(models)}")

    write_json(models)
    write_pages_html()
    write_github_actions()
    update_readme(models, max_rows=cfg.get("readme_max_rows", 200))

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
