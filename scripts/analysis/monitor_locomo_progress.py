import argparse
import json
import os
import pickle
import re
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "locomo10.json"
DEFAULT_CACHE_DIR = ROOT / "cached_memories_robust_deepseek_deepseek-chat_locomo10"
DEFAULT_LOG = ROOT / "logs" / "resume2_locomo10_deepseek_k10_gpu_stderr.log"
DEFAULT_RESULT = ROOT / "results_deepseek_locomo10_k10_gpu.json"


def load_dataset_stats(dataset_path: Path):
    with dataset_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    stats = []
    total_turns = 0
    total_qas = 0
    for idx, sample in enumerate(samples):
        conversation = sample.get("conversation", {})
        turn_count = 0
        for key, value in conversation.items():
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
                turn_count += len(value)
        qa_count = len(sample.get("qa", []))
        total_turns += turn_count
        total_qas += qa_count
        stats.append({"sample": idx, "turns": turn_count, "qas": qa_count})
    return stats, total_turns, total_qas


def tail_lines(path: Path, max_lines: int = 120):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def parse_log(log_lines):
    memory_progress = {}
    cached_samples = set()
    current_sample = None
    processed_questions = 0
    latest_question = None

    progress_re = re.compile(r"Sample (\d+) memory build progress: (\d+)/(\d+) turns, (\d+) memories")
    cached_re = re.compile(r"Successfully cached (\d+) memories")
    processing_re = re.compile(r"Processing sample (\d+)/(\d+)")
    processed_q_re = re.compile(r"Processed (\d+) questions")
    question_re = re.compile(r"Question (\d+):")

    last_cache_sample = None
    for line in log_lines:
        progress_match = progress_re.search(line)
        if progress_match:
            sample = int(progress_match.group(1))
            memory_progress[sample] = {
                "processed_turns": int(progress_match.group(2)),
                "total_turns": int(progress_match.group(3)),
                "memories": int(progress_match.group(4)),
            }
            last_cache_sample = sample
            current_sample = sample

        if cached_re.search(line) and last_cache_sample is not None:
            cached_samples.add(last_cache_sample)

        processing_match = processing_re.search(line)
        if processing_match:
            current_sample = int(processing_match.group(1)) - 1

        processed_q_match = processed_q_re.search(line)
        if processed_q_match:
            processed_questions = max(processed_questions, int(processed_q_match.group(1)))

        question_match = question_re.search(line)
        if question_match:
            latest_question = int(question_match.group(1))
            processed_questions = max(processed_questions, latest_question)

    return {
        "memory_progress": memory_progress,
        "cached_samples": sorted(cached_samples),
        "current_sample": current_sample,
        "processed_questions": processed_questions,
        "latest_question": latest_question,
    }


def file_info(path: Path):
    if not path.exists():
        return {"exists": False, "size": 0, "mtime": None}
    stat = path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def cache_status(cache_dir: Path, sample_count: int):
    statuses = []
    for idx in range(sample_count):
        final_memory = cache_dir / f"memory_cache_sample_{idx}.pkl"
        partial_memory = cache_dir / f"partial_memory_cache_sample_{idx}.pkl"
        status = "pending"
        if final_memory.exists():
            status = "cached"
        elif partial_memory.exists():
            status = "building"
        partial_processed_turns = 0
        partial_memories = 0
        if partial_memory.exists():
            try:
                with partial_memory.open("rb") as f:
                    partial_payload = pickle.load(f)
                partial_processed_turns = int(partial_payload.get("processed_turns", 0) or 0)
                partial_memories = len(partial_payload.get("memories", {}) or {})
            except Exception:
                partial_processed_turns = 0
                partial_memories = 0
        statuses.append({
            "sample": idx,
            "status": status,
            "final_cache": file_info(final_memory),
            "partial_cache": file_info(partial_memory),
            "partial_processed_turns": partial_processed_turns,
            "partial_memories": partial_memories,
        })
    return statuses


def gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip()}
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {"available": False, "raw": first}
    return {
        "available": True,
        "name": parts[0],
        "utilization": int(parts[1]),
        "memory_used": int(parts[2]),
        "memory_total": int(parts[3]),
    }


def load_result_summary(result_path: Path):
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    aggregate = data.get("aggregate_metrics", {})
    overall = aggregate.get("overall", {})
    compact = {}
    for key in ("exact_match", "f1", "rouge1_f", "rougeL_f", "bert_f1", "sbert_similarity"):
        value = overall.get(key, {})
        if isinstance(value, dict) and "mean" in value:
            compact[key] = value["mean"]
    return {
        "total_questions": data.get("total_questions"),
        "completed_samples": data.get("completed_samples"),
        "target_samples": data.get("target_samples"),
        "is_partial": data.get("is_partial"),
        "memory_cache": data.get("memory_cache"),
        "api_usage": data.get("api_usage"),
        "metrics": compact,
    }


def build_status(dataset_path: Path, cache_dir: Path, log_path: Path, result_path: Path):
    sample_stats, total_turns, total_qas = load_dataset_stats(dataset_path)
    log_lines = tail_lines(log_path)
    parsed = parse_log(log_lines)
    caches = cache_status(cache_dir, len(sample_stats))

    completed_turns = 0
    sample_cards = []
    for stat in sample_stats:
        idx = stat["sample"]
        cache = caches[idx]
        progress = parsed["memory_progress"].get(idx, {})
        if cache["status"] == "cached":
            processed_turns = stat["turns"]
        else:
            processed_turns = progress.get("processed_turns", cache.get("partial_processed_turns", 0))
        completed_turns += min(processed_turns, stat["turns"])
        sample_cards.append({
            **stat,
            **cache,
            "processed_turns": processed_turns,
            "memories": progress.get(
                "memories",
                processed_turns if cache["status"] == "cached" else cache.get("partial_memories", 0),
            ),
        })

    result_summary = load_result_summary(result_path)
    processed_questions = parsed["processed_questions"]
    if result_summary and result_summary.get("total_questions"):
        processed_questions = max(processed_questions, int(result_summary["total_questions"]))

    return {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(dataset_path),
        "cache_dir": str(cache_dir),
        "log": file_info(log_path),
        "result": file_info(result_path),
        "current_sample": parsed["current_sample"],
        "total_samples": len(sample_stats),
        "total_turns": total_turns,
        "completed_turns": completed_turns,
        "total_qas": total_qas,
        "processed_questions": processed_questions,
        "samples": sample_cards,
        "recent_log": log_lines[-45:],
        "gpu": gpu_status(),
        "result_summary": result_summary,
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LoCoMo Experiment Progress</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1d2630;
      --muted: #66717f;
      --line: #d9dee7;
      --accent: #1976d2;
      --good: #16845b;
      --warn: #b26a00;
      --pending: #9aa4b2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 "Segoe UI", Arial, sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; }
    main { padding: 20px 24px 28px; max-width: 1280px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .value { font-size: 24px; font-weight: 650; }
    .bar {
      height: 9px;
      border-radius: 999px;
      background: #e8ecf2;
      overflow: hidden;
      margin-top: 10px;
    }
    .fill { height: 100%; background: var(--accent); width: 0%; transition: width .25s ease; }
    .samples {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .sample-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
    .sample-title { font-weight: 650; }
    .status {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .status.cached { color: var(--good); border-color: #9fd5bf; background: #edf8f3; }
    .status.building { color: var(--warn); border-color: #e6c07a; background: #fff7e8; }
    .meta { color: var(--muted); font-size: 12px; margin-top: 8px; display: flex; gap: 12px; flex-wrap: wrap; }
    pre {
      margin: 0;
      max-height: 430px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #101820;
      color: #d9e4ef;
      border-radius: 8px;
      padding: 14px;
      font-size: 12px;
    }
    section { margin-top: 18px; }
    .section-title { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px; }
    h2 { margin: 0; font-size: 16px; }
    .small { color: var(--muted); font-size: 12px; }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
    @media (max-width: 900px) {
      .grid, .samples, .metrics { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>LoCoMo DeepSeek k=10 Experiment Progress</h1>
      <div class="small" id="paths"></div>
    </div>
    <div class="small" id="updated">正在读取...</div>
  </header>
  <main>
    <div class="grid">
      <div class="panel">
        <div class="label">Memory turns</div>
        <div class="value" id="turnValue">0 / 0</div>
        <div class="bar"><div class="fill" id="turnBar"></div></div>
      </div>
      <div class="panel">
        <div class="label">QA answered</div>
        <div class="value" id="qaValue">0 / 0</div>
        <div class="bar"><div class="fill" id="qaBar"></div></div>
      </div>
      <div class="panel">
        <div class="label">Current sample</div>
        <div class="value" id="sampleValue">-</div>
        <div class="small" id="resultState"></div>
      </div>
      <div class="panel">
        <div class="label">GPU</div>
        <div class="value" id="gpuValue">-</div>
        <div class="small" id="gpuDetail"></div>
      </div>
    </div>

    <section>
      <div class="section-title">
        <h2>Sample 缓存与构建进度</h2>
        <div class="small">cached 表示该 sample 的记忆可复用；building 表示 partial checkpoint 已写入</div>
      </div>
      <div class="samples" id="samples"></div>
    </section>

    <section id="resultSection" style="display:none">
      <div class="section-title"><h2>已生成结果摘要</h2><div class="small">来自结果 JSON</div></div>
      <div class="metrics" id="metrics"></div>
    </section>

    <section>
      <div class="section-title"><h2>最近日志</h2><div class="small" id="logInfo"></div></div>
      <pre id="logs"></pre>
    </section>
  </main>
  <script>
    function pct(done, total) {
      if (!total) return 0;
      return Math.max(0, Math.min(100, done / total * 100));
    }
    function fmtPct(done, total) {
      return pct(done, total).toFixed(1) + "%";
    }
    function setBar(id, done, total) {
      document.getElementById(id).style.width = pct(done, total) + "%";
    }
    async function refresh() {
      const res = await fetch("/api/status", { cache: "no-store" });
      const data = await res.json();
      document.getElementById("updated").textContent = "更新时间 " + data.updated_at;
      document.getElementById("paths").textContent = data.result.exists ? "结果文件已生成" : "结果文件未生成，实验仍在进行或等待";
      document.getElementById("turnValue").textContent = data.completed_turns + " / " + data.total_turns;
      document.getElementById("qaValue").textContent = data.processed_questions + " / " + data.total_qas;
      document.getElementById("sampleValue").textContent = data.current_sample === null ? "-" : (data.current_sample + 1) + " / " + data.total_samples;
      document.getElementById("resultState").textContent = data.result.exists ? "results_deepseek_locomo10_k10_gpu.json" : "running / no final result yet";
      setBar("turnBar", data.completed_turns, data.total_turns);
      setBar("qaBar", data.processed_questions, data.total_qas);

      if (data.gpu.available) {
        document.getElementById("gpuValue").textContent = data.gpu.utilization + "%";
        document.getElementById("gpuDetail").textContent = data.gpu.name + " | " + data.gpu.memory_used + " / " + data.gpu.memory_total + " MiB";
      } else {
        document.getElementById("gpuValue").textContent = "-";
        document.getElementById("gpuDetail").textContent = data.gpu.error || "nvidia-smi unavailable";
      }

      const samples = document.getElementById("samples");
      samples.innerHTML = "";
      data.samples.forEach(sample => {
        const card = document.createElement("div");
        card.className = "panel";
        const percent = fmtPct(sample.processed_turns, sample.turns);
        card.innerHTML = `
          <div class="sample-head">
            <div class="sample-title">Sample ${sample.sample}</div>
            <div class="status ${sample.status}">${sample.status}</div>
          </div>
          <div class="bar"><div class="fill" style="width:${pct(sample.processed_turns, sample.turns)}%"></div></div>
          <div class="meta">
            <span>turns ${sample.processed_turns}/${sample.turns} (${percent})</span>
            <span>memories ${sample.memories}</span>
            <span>QA ${sample.qas}</span>
          </div>
        `;
        samples.appendChild(card);
      });

      document.getElementById("logInfo").textContent = data.log.exists ? ("日志更新时间 " + data.log.mtime) : "日志不存在";
      document.getElementById("logs").textContent = data.recent_log.join("\n");

      const resultSection = document.getElementById("resultSection");
      const metrics = document.getElementById("metrics");
      metrics.innerHTML = "";
      if (data.result_summary) {
        resultSection.style.display = "";
        Object.entries(data.result_summary.metrics || {}).forEach(([key, value]) => {
          const box = document.createElement("div");
          box.className = "panel";
          box.innerHTML = `<div class="label">${key}</div><div class="value">${Number(value).toFixed(4)}</div>`;
          metrics.appendChild(box);
        });
      } else {
        resultSection.style.display = "none";
      }
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    dataset_path = DEFAULT_DATASET
    cache_dir = DEFAULT_CACHE_DIR
    log_path = DEFAULT_LOG
    result_path = DEFAULT_RESULT

    def _send(self, code, content, content_type):
        encoded = content.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            try:
                status = build_status(self.dataset_path, self.cache_dir, self.log_path, self.result_path)
                self._send(200, json.dumps(status, ensure_ascii=False), "application/json; charset=utf-8")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}, ensure_ascii=False), "application/json; charset=utf-8")
            return
        self._send(404, "Not found", "text/plain; charset=utf-8")

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Serve a local LoCoMo experiment progress dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    args = parser.parse_args()

    Handler.dataset_path = Path(args.dataset)
    Handler.cache_dir = Path(args.cache_dir)
    Handler.log_path = Path(args.log)
    Handler.result_path = Path(args.result)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LoCoMo progress dashboard: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
