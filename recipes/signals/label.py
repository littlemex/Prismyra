"""self_solve labels: the base model, used as a normal chat LLM (thinking off), answers each problem k times.
Label = fraction of the k samples judged correct. Also one greedy sample. Checkers are deliberately simple and
their failure modes are counted (unparsed answers) rather than hidden."""

import asyncio, json, re, sys, string
import httpx

URL, OUT, K = sys.argv[1], sys.argv[2], int(sys.argv[3])  # usage: label.py <chat URL> <out.jsonl> <k> <problems.json>
probs = json.load(open(sys.argv[4]))


def last(pattern, text):
    m = re.findall(pattern, text, re.S | re.I)
    return m[-1].strip() if m else None


def boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(t) and d:
        d += {"{": 1, "}": -1}.get(t[j], 0)
        j += 1
    return t[i + 7 : j - 1]


def norm(s):
    s = s.lower().strip().strip(".")
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", s).split()


def normm(s):
    s = re.sub(r"\\[dt]frac", r"\\frac", s)  # \dfrac and \tfrac are \frac, backslash kept
    s = re.sub(r"\\(left|right|!|,|;)", "", s)
    return s.replace(" ", "").replace("^\\circ", "").replace("\\%", "").replace("$", "").strip(".")


def judge(p, text):
    if p["kind"] == "boxed":
        g = boxed(text)
        return None if g is None else normm(g) == normm(p["answer"])
    a = last(r"Answer:\s*(.+?)(?:\n|$)", text)
    if a is None:
        return None
    if p["kind"] == "letter":  # ARC also labels some options 1-4
        m = re.match(r"\(?([A-J1-9])\)?", a.strip())
        return bool(m) and m.group(1) == p["answer"]
    if p["kind"] == "number":
        n = re.findall(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
        return bool(n) and abs(float(n[-1]) - float(p["answer"])) < 1e-6
    if p["kind"] == "alias":
        na = norm(a)
        return any(norm(g) == na or (norm(g) and " ".join(norm(g)) in " ".join(na)) for g in p["answer"])
    if re.fullmatch(r"-?\d+(?:\.\d+)?", p["answer"].strip()):  # arithmetic answers compare as numbers, sign kept
        n = re.findall(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
        return bool(n) and abs(float(n[-1]) - float(p["answer"])) < 1e-6
    return norm(a) == norm(p["answer"])


async def one(client, p, sem):
    async with sem:
        body = {
            "model": "qwen36",
            "messages": [{"role": "user", "content": p["prompt"]}],
            "max_tokens": 1536,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        g = await client.post(URL, json={**body, "temperature": 0.0, "n": 1})
        s = await client.post(URL, json={**body, "temperature": 0.7, "top_p": 0.8, "n": K})
        gt = g.json()["choices"][0]["message"]["content"]
        st = [c["message"]["content"] for c in s.json()["choices"]]
        jg = judge(p, gt)
        js = [judge(p, t) for t in st]
        return {
            "id": p["id"],
            "family": p["family"],
            "sub": p["sub"],
            "greedy": jg,
            "samples": js,
            "rate": sum(bool(x) for x in js) / K,
            "unparsed": sum(x is None for x in js) + (jg is None),
            "greedy_text": gt[-400:],
        }


async def main():
    sem = asyncio.Semaphore(48)
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(OUT)}
    except FileNotFoundError:
        pass
    f = open(OUT, "a")
    async with httpx.AsyncClient(timeout=600) as client:
        tasks = [one(client, p, sem) for p in probs if p["id"] not in done]
        for n, t in enumerate(asyncio.as_completed(tasks)):
            try:
                r = await t
            except Exception as e:
                print("ERR", e, flush=True)
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 200 == 0:
                print(n, flush=True)


asyncio.run(main())
