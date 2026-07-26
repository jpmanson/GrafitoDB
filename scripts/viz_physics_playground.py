#!/usr/bin/env python3
"""Local playground to debug PyVis physics / spacing (no Colab).

Builds a small DocumentIngestor graph (section tree + chunks + NEXT_PASSAGE)
and writes several HTML variants so you can open them in a browser and compare.

Usage (from repo root)::

    uv run python scripts/viz_physics_playground.py
    uv run python scripts/viz_physics_playground.py --open          # open all in browser
    uv run python scripts/viz_physics_playground.py --only wide     # one variant
    uv run python scripts/viz_physics_playground.py --out /tmp/viz

What we learned
---------------
vis.js only uses ``physics.repulsion.*`` when ``physics.solver == "repulsion"``.
Passing ``{"physics": {"repulsion": {"nodeDistance": 400}}}`` via
``set_options`` without the solver leaves the default Barnes-Hut attractor
running — springs feel "magnetic" and distances look ignored. Grafito's
``save_pyvis_html`` now applies presets through ``Network.repulsion()`` so the
solver is set correctly.
"""

from __future__ import annotations

import argparse
import re
import webbrowser
from pathlib import Path

from grafito import GrafitoDatabase
from grafito.document import DocumentIngestor, MarkdownChunker, RecursiveChunker
from grafito.integrations import save_pyvis_html

SAMPLE_MD = """# Common use cases and applications for AI agents

Intro paragraph about why use cases matter for agent design and ROI.

## Customer support

Agents that triage tickets and draft replies with tool access to CRM.

## Data analysis

Agents that pull metrics, plot trends, and summarize for stakeholders.

## Document processing

Agents that extract fields from PDFs and route them into workflows.
"""


def _label(node_id, attrs):
    p, labels = attrs.get("properties") or {}, attrs.get("labels") or []
    if "Document" in labels:
        return f"#{node_id} PDF"
    if "DocumentVersion" in labels:
        return f"#{node_id} version gen={p.get('generation', '?')}"
    if "Section" in labels:
        return f"#{node_id} H{p.get('level', '?')} {(p.get('title') or '')[:28]}"
    if "Chunk" in labels:
        snip = (p.get("text") or "")[:22].replace("\n", " ")
        return f"#{node_id} s={p.get('global_seq', '?')} {snip}"
    return f"#{node_id}"


def build_demo_graph():
    db = GrafitoDatabase(":memory:")
    ing = DocumentIngestor(
        db,
        chunker=MarkdownChunker(
            max_chars=180,
            overlap=0,
            overflow_chunker=RecursiveChunker(max_size=180, overlap=0),
        ),
        hierarchy="auto",
        write_next_passage=True,
    )
    result = ing.ingest(
        SAMPLE_MD,
        document_key="demo/use-cases",
        title="Demo use cases",
        embed=False,
    )
    parent = db.match_nodes(labels=["Document"], properties={"document_key": "demo/use-cases"}, limit=1)[0]
    # Focus: root section + children + chunks (same idea as notebook §5)
    roots = [
        n
        for n in db.match_nodes(properties={"managed_by": "grafito.document", "role": "section"})
        if n.properties.get("owner_document_id") == parent.id
        and (n.properties.get("title") or "").startswith("Common use cases")
    ]
    if not roots:
        roots = [
            n
            for n in db.match_nodes(properties={"managed_by": "grafito.document", "role": "section"})
            if n.properties.get("owner_document_id") == parent.id
        ][:1]
    sec = roots[0]
    ids = {sec.id}
    stack = [sec.id]
    while stack:
        sid = stack.pop()
        for child in db.get_neighbors(sid, direction="outgoing", rel_type=ing.has_section_rel):
            if child.id not in ids:
                ids.add(child.id)
                stack.append(child.id)
        for chunk in db.get_neighbors(sid, direction="outgoing", rel_type=ing.has_passage_rel):
            ids.add(chunk.id)

    G = db.to_networkx().subgraph(ids).copy()
    # color-ish attrs for readability
    for nid in G.nodes:
        labels = G.nodes[nid].get("labels") or []
        props = dict(G.nodes[nid].get("properties") or {})
        if "Section" in labels:
            props["_c"] = "#e9c46a"
        elif "Chunk" in labels:
            props["_c"] = "#f4a261"
        else:
            props["_c"] = "#8ecae6"
        G.nodes[nid]["properties"] = props

    meta = {
        "n_sections": result.n_sections,
        "n_passages": result.n_passages,
        "focus": f"#{sec.id} {sec.properties.get('title')}",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "edge_types": {},
    }
    for _, _, d in G.edges(data=True):
        t = d.get("type") or "?"
        meta["edge_types"][t] = meta["edge_types"].get(t, 0) + 1
    db.close()
    return G, meta


VARIANTS: dict[str, object] = {
    # What the library preset meant to do (now applied via Network.repulsion)
    "spread_preset": "spread",
    "compact_preset": "compact",
    # Explicit distances (solver set correctly by save_pyvis_html)
    "wide": {
        "physics": {
            "solver": "repulsion",
            "repulsion": {
                "nodeDistance": 350,
                "springLength": 300,
                "springConstant": 0.01,
                "centralGravity": 0.05,
                "damping": 0.09,
            },
        }
    },
    "extra_wide": {
        "physics": {
            "solver": "repulsion",
            "repulsion": {
                "nodeDistance": 500,
                "springLength": 450,
                "springConstant": 0.008,
                "centralGravity": 0.02,
                "damping": 0.1,
            },
        }
    },
    # Sent via save_pyvis_html — library now *infers* solver=repulsion, so this
    # one is fixed on purpose (same as wide-ish). See broken_raw_set_options below.
    "inferred_solver": {
        "physics": {
            "repulsion": {
                "nodeDistance": 500,
                "springLength": 450,
                "springConstant": 0.008,
            }
        }
    },
}


def _extract_solver_info(html: str) -> str:
    m = re.search(r"var options = (\{.*?\});", html, re.S)
    if not m:
        return "options: not found"
    import json

    try:
        opts = json.loads(m.group(1))
    except json.JSONDecodeError:
        return "options: unparseable"
    phys = opts.get("physics") or {}
    solver = phys.get("solver", "(default/missing)")
    if "repulsion" in phys:
        r = phys["repulsion"]
        return (
            f"solver={solver}  nodeDistance={r.get('nodeDistance')}  "
            f"springLength={r.get('springLength')}  centralGravity={r.get('centralGravity')}"
        )
    if "barnesHut" in phys:
        b = phys["barnesHut"]
        return f"solver={solver}  barnesHut springLength={b.get('springLength')}"
    return f"solver={solver}  keys={list(phys.keys())}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("tmp/viz_physics_playground"))
    ap.add_argument("--open", action="store_true", help="Open generated HTML files in the browser")
    ap.add_argument("--only", choices=sorted(VARIANTS), help="Generate a single variant")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    G, meta = build_demo_graph()
    print("Demo graph:", meta)

    variants = {args.only: VARIANTS[args.only]} if args.only else dict(VARIANTS)
    index_rows = []
    for name, physics in variants.items():
        path = args.out / f"{name}.html"
        save_pyvis_html(
            G,
            path=str(path),
            notebook=False,
            directed=True,
            color_by_label=False,
            node_color_attr="_c",
            label_fn=_label,
            physics=physics,
            height="700px",
            width="100%",
            bgcolor="#fff",
            font_color="#222",
            cdn_resources="in_line",
        )
        html = path.read_text(encoding="utf-8")
        info = _extract_solver_info(html)
        print(f"  wrote {path}  ({info})")
        index_rows.append((name, path, info))
        if args.open:
            webbrowser.open(path.resolve().as_uri())

    # Reproduce the historical bug: raw set_options without solver (bypass helper).
    if args.only is None or args.only == "broken_raw_set_options":
        from pyvis.network import Network
        import json as _json

        path = args.out / "broken_raw_set_options.html"
        net = Network(directed=True, height="700px", width="100%", bgcolor="#fff",
                      font_color="#222", cdn_resources="in_line")
        for nid, attrs in G.nodes(data=True):
            net.add_node(
                nid,
                label=_label(nid, attrs),
                color=(attrs.get("properties") or {}).get("_c", "#8ecae6"),
            )
        for u, v, d in G.edges(data=True):
            net.add_edge(u, v, label=d.get("type") or "")
        # Classic bug: repulsion numbers, no solver → Barnes-Hut still runs.
        net.set_options(
            _json.dumps(
                {
                    "physics": {
                        "repulsion": {
                            "nodeDistance": 500,
                            "springLength": 450,
                            "springConstant": 0.008,
                        }
                    }
                }
            )
        )
        path.write_text(net.generate_html(notebook=False), encoding="utf-8")
        info = _extract_solver_info(path.read_text(encoding="utf-8"))
        print(f"  wrote {path}  ({info})  ← historical bug")
        index_rows.append(("broken_raw_set_options", path, info + " [BUG repro]"))
        if args.open:
            webbrowser.open(path.resolve().as_uri())

    # Index page
    index = args.out / "index.html"
    links = "\n".join(
        f'<li><a href="{p.name}" target="_blank"><b>{name}</b></a> — <code>{info}</code></li>'
        for name, p, info in index_rows
    )
    index.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>viz physics playground</title>
<style>
 body {{ font: 15px/1.45 system-ui,sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
 code {{ background: #f4f4f4; padding: 0.1em 0.35em; border-radius: 4px; }}
 li {{ margin: 0.6rem 0; }}
</style></head><body>
<h1>Grafito PyVis physics playground</h1>
<p>Focus: <b>{meta['focus']}</b> — {meta['nodes']} nodes, edges {meta['edge_types']}</p>
<p>Compare variants. <b>broken_no_solver</b> is the old bug (params without
<code>solver: repulsion</code>); distances look ignored because Barnes-Hut still runs.</p>
<ul>
{links}
</ul>
</body></html>
""",
        encoding="utf-8",
    )
    print(f"\nIndex: {index.resolve()}")
    if args.open:
        webbrowser.open(index.resolve().as_uri())
    else:
        print(f"Open: open {index}")


if __name__ == "__main__":
    main()
