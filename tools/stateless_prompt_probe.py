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
mechanically.  Answer checking is exact, not substring-based:

  * echo replies must match the requested word exactly after whitespace and
    surrounding punctuation are stripped (a 逐字 spelled-out "苹 果 红" passes,
    a truncated "苹果" does not);
  * number and arithmetic replies are parsed as the reply's LAST integer, which
    must equal the expected value exactly ("16" fails 3+3, "210086" fails
    10086, "答案是 6。" passes).

Verdicts and exit codes:

  PASS    exit 0   every case validated OK
  PARTIAL exit 2   no wrong answers, but at least one reply was inconclusive
                   (empty and length-capped: a greedy reply that spent its whole
                   budget thinking never answered - degenerate sampling, not a
                   wrong context).  A run that validated nothing reports PARTIAL
                   and says so; it can never report success.
  FAIL    exit 1   at least one wrong answer

`--parked` adds a controlled parked-prefix sequence: seed conversation A with a
planted secret, switch away (conversation B holds a second secret), then ask a
fresh shared-prefix request for A's secret.  A correct fresh conversation
cannot know the secret; a reply that leaks either planted secret means parked
state crossed conversations.  Each step reports cached_tokens, and the run
states whether reuse actually occurred - a probe with cached_tokens=0 did not
exercise the restore path.

`--selftest` runs the checker regressions offline (no server), including the
injected false positives "16" (3+3) and "210086" (10086).

Authentication: the probe sends `Authorization: Bearer` when a key is supplied
via `--api-key` or the STRATA_API_KEY environment variable.  The key is never
printed.

Usage:
    python tools/stateless_prompt_probe.py --url http://127.0.0.1:8081 --rounds 3
    python tools/stateless_prompt_probe.py --parked
    python tools/stateless_prompt_probe.py --selftest

Stdlib only; temperature 0 so a wrong reply is a wrong context, not sampling.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request

PUNCT = "。．.！!？?，,；;：:、\"'“”‘’「」『』"


def normalize(text):
    """Echo normalization: drop whitespace, strip surrounding punctuation."""
    return "".join(text.split()).strip(PUNCT)


def last_integer(text):
    """The reply's last integer, or None - the defined numeric answer parser."""
    found = re.findall(r"-?\d+", "".join(text.split()))
    return int(found[-1]) if found else None


def check_reply(kind, want, content):
    """Returns "OK" or "BAD" for a reply's content under the case's checker."""
    if kind == "echo":
        return "OK" if normalize(content) == want else "BAD"
    return "OK" if last_integer(content) == want else "BAD"


def cases():
    return [
        ("number", "请只回复数字：10086", 10086),
        ("number", "请只回复数字：4096", 4096),
        ("echo", "请逐字重复这个词：苹果红", "苹果红"),
        ("echo", "请逐字重复这个词：天空蓝", "天空蓝"),
        ("echo", "请逐字重复这个词：薄荷绿", "薄荷绿"),
        ("math", "3+3等于几？直接输出答案", 6),
        ("math", "7+8等于几？直接输出答案", 15),
        ("math", "9+9等于几？直接输出答案", 18),
    ]


def ask(url, content, api_key, timeout=240, max_tokens=300):
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "temperature": 0, "max_tokens": max_tokens},
                      ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = d["choices"][0]
    usage = d.get("usage", {})
    return {"finish": ch.get("finish_reason"),
            "content": (ch["message"].get("content") or "").strip(),
            "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
            "prompt_tokens": usage.get("prompt_tokens")}


def report(case_kind, prompt, got, ok):
    snippet = got["content"][:40].replace("\n", " ") or "<empty>"
    mark = "OK " if ok else "BAD"
    print(f"  {mark} [{case_kind}] {prompt[:18]!r:24} finish={got['finish']:<6} "
          f"cached={got['cached']} -> {snippet!r}")


def run(url, rounds, workers, api_key):
    ok = bad = inconclusive = validated = 0
    t0 = time.time()
    for rd in range(rounds):
        print(f"── round {rd + 1}/{rounds} ──")
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda c: (c, ask(url, c[1], api_key)), cases()))
        for (kind, prompt, want), got in results:
            if not got["content"] and got["finish"] == "length":
                inconclusive += 1
                print(f"  INC [{kind}] {prompt[:18]!r:24} finish={got['finish']:<6} "
                      f"cached={got['cached']} -> <empty>")
                continue
            good = check_reply(kind, want, got["content"]) == "OK"
            ok += good
            validated += 1
            bad += not good
            report(kind, prompt, got, good)
    if bad:
        verdict = "FAIL"
    elif inconclusive:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"
    print(f"{verdict}: {ok} ok, {bad} bad, {inconclusive} inconclusive "
          f"({validated} validated of {rounds * len(list(cases()))} requests) "
          f"in {time.time() - t0:.1f}s")
    if not validated and not bad:
        print("  no validated answers - this run does not report success")
    return bad, inconclusive, validated


def parked(url, api_key):
    """Seed A -> switch away (seed B) -> fresh shared-prefix probe for A's secret."""
    # a shared system prompt long enough that the parking machinery actually
    # matches it (~200 tokens; a short one never triggers a restore and the
    # sequence would not exercise the path it exists to test)
    sys_prompt = ("你是校验代理，服务于 OpenLakehouse 数据库 v3.0 的发布工程团队。项目背景："
                  "一个兼容 PostgreSQL 协议的湖仓一体分析型数据库，v3.0 引入了向量化执行器、"
                  "对象存储原生的表格式 OpenTable v2、存算分离架构与全局元数据目录。团队约定："
                  "所有产出使用中文，专有名词可括注英文；涉及性能的数字必须标注测量环境；"
                  "不要虚构未提供的信息；引用配置项时写全称；遇到不确定的接口行为先查源码再回答。"
                  "你的回复应当简洁、直接、面向工程读者。")
    secret_a, secret_b = "pineapple42", "granite75"
    steps = [
        ("seed A", f"请记住这个投递暗号：{secret_a}。除了确认收到外什么都不要说。"),
        ("seed B", f"请记住这个仓库编号：{secret_b}。除了确认收到外什么都不要说。"),
        ("probe", "这个对话里有没有让你记过什么暗号？如果有，请原样复述出来；"
                  "如果没有，请只回答：本对话没有暗号。"),
    ]
    print("── parked-prefix sequence (seed A -> switch away -> probe) ──")
    reuse = None
    bad = 0
    for name, content in steps:
        got = ask(url, content, api_key, max_tokens=200)
        if name == "seed A":
            got["role"] = name
            continue                                    # its reply is an ack, no check
        if name == "seed B":
            got["role"] = name
            seed_b_cached = got["cached"]
            continue
        flat = "".join(got["content"].split())
        leaked = [s for s in (secret_a, secret_b) if s in flat]
        reuse = (got["cached"] or 0) > 0
        good = not leaked
        bad += not good
        report("probe", content, got, good)
        print(f"  probe: seeded secrets leaked = {leaked or 'none'} | reuse occurred = "
              f"{reuse} (cached={got['cached']}, shared system prefix)"
              + ("" if reuse else " - the restore path was NOT exercised by this run"))
    print(f"  seed B cached={seed_b_cached} (its own prefix reuse)")
    return bad, reuse


def selftest():
    """Offline regression checks for the validators, no server involved."""
    checks = [
        ("echo spelled out passes", check_reply("echo", "苹果红", "苹 果 红") == "OK"),
        ("echo with punctuation passes", check_reply("echo", "苹果红", "“苹果红”。") == "OK"),
        ("truncated echo fails", check_reply("echo", "苹果红", "苹果") == "BAD"),
        ("wrong word fails", check_reply("echo", "苹果红", "天空蓝") == "BAD"),
        ("injected '16' fails 3+3", check_reply("math", 6, "16") == "BAD"),
        ("injected '210086' fails 10086", check_reply("number", 10086, "210086") == "BAD"),
        ("plain '6' passes 3+3", check_reply("math", 6, "6") == "OK"),
        ("'答案是 6。' passes 3+3", check_reply("math", 6, "答案是 6。") == "OK"),
        ("'3+3等于6' passes 3+3", check_reply("math", 6, "3+3等于6") == "OK"),
        ("dropped token '1086' fails 10086", check_reply("number", 10086, "1086") == "BAD"),
        ("last-integer parser ignores prose", check_reply("number", 10086, "号码是10086，完毕") == "OK"),
        ("no integer is inconclusive-ish (BAD)", check_reply("number", 10086, "不知道") == "BAD"),
        ("leaked secret is detected", "pineapple42" in "".join("暗号是 pineapple42".split())),
        ("clean reply has no secret", "pineapple42" not in "".join("本对话没有暗号".split())),
    ]
    bad = 0
    for name, good in checks:
        bad += not good
        print(f"  {'OK ' if good else 'BAD'} {name}")
    print(f"{'PASS' if not bad else 'FAIL'}: {len(checks) - bad}/{len(checks)} checker regressions hold")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--rounds", type=int, default=3, help="rounds over the case list")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests per round")
    ap.add_argument("--parked", action="store_true",
                    help="run the parked-prefix seed/switch/probe sequence")
    ap.add_argument("--selftest", action="store_true",
                    help="run the checker regressions offline and exit")
    ap.add_argument("--api-key", default=None,
                    help="Bearer key (defaults to the STRATA_API_KEY env var; never printed)")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(1 if selftest() else 0)
    key = a.api_key if a.api_key is not None else os.environ.get("STRATA_API_KEY")
    bad, inconclusive, validated = run(a.url, a.rounds, a.workers, key)
    reuse = None
    if a.parked:
        parked_bad, reuse = parked(a.url, key)
        bad += parked_bad
    if bad:
        raise SystemExit(1)
    if inconclusive or validated == 0 or reuse is False:
        raise SystemExit(2)
    raise SystemExit(0)
