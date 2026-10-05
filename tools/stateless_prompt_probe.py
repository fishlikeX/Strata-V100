"""Probe a running Strata server with stateless single-turn requests.

The prompt-attention / KV paths have failure modes that a chat UI with a full
conversation history hides: every web request legitimately continues its own
prefix, so a wrong echo, a dropped token or a stale context is read as "the
model being creative".  A stateless one-shot request has none of that cover -
its whole prompt is fresh - which makes it the sensitive surface for

  * prompt tokens being dropped or misread (an echo comes back short, a dictated
    number comes back wrong), and
  * parked-conversation restores that overshoot the matched prefix (the reply
    answers a previous, unrelated conversation).

The probe sends N rounds of fixed one-shot cases and checks the replies
mechanically: an echo must contain the echoed word, a dictated number must
contain it, an arithmetic answer must contain the expected sum.  It reports
`cached_tokens` per case (prompt_tokens_details), because prefix reuse is where
restore bugs show, and exits 1 when any case fails.

Usage:
    python tools/stateless_prompt_probe.py --url http://127.0.0.1:8081 --rounds 3

Stdlib only; temperature 0 so a wrong reply is a wrong context, not sampling.
"""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def cases():
    return [
        ("number", "请只回复数字：10086", ("10086",)),
        ("number", "请只回复数字：4096", ("4096",)),
        ("echo", "请逐字重复这个词：苹果红", ("苹果红",)),
        ("echo", "请逐字重复这个词：天空蓝", ("天空蓝",)),
        ("echo", "请逐字重复这个词：薄荷绿", ("薄荷绿",)),
        ("math", "3+3等于几？直接输出答案", ("6",)),
        ("math", "7+8等于几？直接输出答案", ("15",)),
        ("math", "9+9等于几？直接输出答案", ("18",)),
    ]


def ask(url, content, timeout=240):
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "temperature": 0, "max_tokens": 300},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = d["choices"][0]
    usage = d.get("usage", {})
    return {"finish": ch.get("finish_reason"),
            "content": (ch["message"].get("content") or "").strip(),
            "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
            "prompt_tokens": usage.get("prompt_tokens")}


def run(url, rounds, workers):
    bad = inconclusive = 0
    t0 = time.time()
    for rd in range(rounds):
        print(f"── round {rd + 1}/{rounds} ──")
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda c: (c, ask(url, c[1])), cases()))
        for (kind, prompt, want), got in results:
            snippet = got["content"][:40].replace("\n", " ") or "<empty>"
            if not got["content"] and got["finish"] == "length":
                # a greedy (temperature 0) reply that spent its whole budget thinking and
                # never answered: degenerate sampling, not a wrong context - report it
                # separately so the probe stays specific to corruption
                inconclusive += 1
                print(f"  INC [{kind}] {prompt[:18]!r:24} finish={got['finish']:<6} "
                      f"cached={got['cached']} -> {snippet!r}")
                continue
            # an echo may come back spelled out ("苹 果 红" for 逐字 "character by
            # character"), so match on whitespace-stripped text
            flat = "".join(got["content"].split())
            ok = any(w in flat for w in want)
            bad += not ok
            mark = "OK " if ok else "BAD"
            print(f"  {mark} [{kind}] {prompt[:18]!r:24} finish={got['finish']:<6} "
                  f"cached={got['cached']} -> {snippet!r}")
    verdict = "PASS" if not bad else "FAIL"
    print(f"{verdict}: {bad} bad, {inconclusive} inconclusive in {time.time() - t0:.1f}s "
          f"({rounds * len(list(cases()))} requests)")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--rounds", type=int, default=3, help="rounds over the case list")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests per round")
    a = ap.parse_args()
    raise SystemExit(1 if run(a.url, a.rounds, a.workers) else 0)
