"""Strip the FP8 routed-expert tensors out of each per-layer shard (pure stdlib, no torch/safetensors
needed): parse the safetensors header, keep only the non-expert keys, re-write a small shard with
freshly computed byte offsets. Mirrors the published 36l NVFP4 repo's layout (layers-N.safetensors
containing only the 22-ish non-expert tensors per layer; the experts live in nvfp4_experts.safetensors
instead).  Usage: repack_32l.py SRC_DIR DST_DIR
"""
import json, os, re, struct, sys

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data_start = 8 + n
    return header, data_start


def strip_layer(src_path, dst_path):
    header, data_start = read_header(src_path)
    meta = header.pop("__metadata__", None)
    keep = {k: v for k, v in header.items() if not EXPERT_RE.search(k)}
    # preserve relative order by original data_offsets start
    ordered = sorted(keep.items(), key=lambda kv: kv[1]["data_offsets"][0])
    new_header = {}
    if meta is not None:
        new_header["__metadata__"] = meta
    cursor = 0
    with open(src_path, "rb") as fin, open(dst_path, "wb") as placeholder:
        pass  # reopen below once header size is known; write data to a temp buffer file instead
    tmp_data_path = dst_path + ".data.tmp"
    with open(src_path, "rb") as fin, open(tmp_data_path, "wb") as fdata:
        for name, info in ordered:
            start, end = info["data_offsets"]
            length = end - start
            fin.seek(data_start + start)
            remaining = length
            while remaining > 0:
                chunk = fin.read(min(remaining, 64 * 1024 * 1024))
                if not chunk:
                    raise IOError(f"short read on {name} in {src_path}")
                fdata.write(chunk)
                remaining -= len(chunk)
            new_header[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [cursor, cursor + length],
            }
            cursor += length
    header_bytes = json.dumps(new_header).encode("utf-8")
    with open(dst_path, "wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        with open(tmp_data_path, "rb") as fdata:
            while True:
                chunk = fdata.read(64 * 1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
    os.remove(tmp_data_path)
    return len(keep), len(header) - (1 if meta is not None else 0)


def main():
    idx_path = os.path.join(SRC, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    layer_files = sorted({v for v in wm.values() if v.startswith("layers-")})
    total_kept = total_src = 0
    for fname in layer_files:
        src_path = os.path.join(SRC, fname)
        dst_path = os.path.join(DST, fname)
        kept, src_n = strip_layer(src_path, dst_path)
        total_kept += kept
        total_src += src_n
        print(fname, "kept", kept, "of", src_n, "src_size", os.path.getsize(src_path),
              "dst_size", os.path.getsize(dst_path), flush=True)
    # copy everything else verbatim (skip the internal .merged marker and the index itself)
    for f in sorted(os.listdir(SRC)):
        src_f = os.path.join(SRC, f)
        if not os.path.isfile(src_f):
            continue
        if f in (".merged", "model.safetensors.index.json"):
            continue
        if f in layer_files:
            continue
        dst_f = os.path.join(DST, f)
        if not os.path.exists(dst_f):
            with open(src_f, "rb") as a, open(dst_f, "wb") as b:
                while True:
                    chunk = a.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    b.write(chunk)
    new_wm = {k: v for k, v in wm.items() if not EXPERT_RE.search(k)}
    json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_wm},
              open(os.path.join(DST, "model.safetensors.index.json"), "w"))
    print("done. total_kept_keys", total_kept, "total_src_keys", total_src)


if __name__ == "__main__":
    main()
