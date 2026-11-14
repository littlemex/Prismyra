"""perigraph's conformance runner adapter for Prismyra's writer (stamper role).

    python <perigraph>/tools/conform.py --adapter "python tools/perigraph_conform.py" --role stamper

Reads one JSON request per line on stdin and answers one per line on stdout, as perigraph's conformance/README.md
defines. Every answer is computed by `prismyra.identity` -- the digests, the identities and the full wire record a
served answer carries -- so the runner checks this writer, not a reimplementation of it.
"""

from __future__ import annotations

import hashlib
import json
import sys

from prismyra import identity as I


def describe(_req: dict) -> dict:
    return {"name": "prismyra (prismyra.identity)", "roles": ["stamper"]}


def stamp(req: dict) -> dict:
    try:
        ck = I.checkpoint_digest([(p, hashlib.sha256(t.encode("utf-8")).hexdigest()) for p, t in req["files"].items()])
        pr = I.precision_digest(req["precision"])
        rt = I.lines_digest(req["runtime"], "==")
        am = I._answer_mode(req["answer_mode"])
    except I.IdentityError as e:
        return {"refused": str(e)}
    st = I.ModelStamp(
        {
            "checkpoint": I.Part(digest=ck),
            "precision": I.Part(digest=pr),
            "runtime": I.Part(digest=rt),
            "answer_mode": am,
        }
    )
    return {
        "checkpoint_digest": ck,
        "precision_digest": pr,
        "runtime_digest": rt,
        "model_identity": st.model_identity,
        "artifact_identity": st.artifact_identity,
        "attributes": st.attributes(),
    }


OPS = {"describe": describe, "stamp": stamp}

for line in sys.stdin:
    req = json.loads(line)
    reply = OPS[req["op"]](req) if req["op"] in OPS else {"error": f"unsupported op {req['op']!r}"}
    print(json.dumps({"id": req["id"], **reply}, ensure_ascii=False), flush=True)
