from __future__ import annotations

import json
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import tyro


@dataclass
class Args:
    annotations_path: str = (
        "/mnt/disk_2/wjz/runs/pi05_subtask_annotations/"
        "pick_and_place_tape_into_cup_full_qwen25vl7b_local_prompt2/annotations.json"
    )
    output_path: str | None = None
    host: str = "0.0.0.0"
    port: int = 8766
    needs_review_only: bool = True


def _needs_review(record: dict) -> bool:
    return record.get("review_status") == "needs_review" or any(
        str(flag).startswith("needs_review") for flag in record.get("flags", [])
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _update_record_boundary(record: dict, boundary_frame: int, note: str = "") -> None:
    num_frames = int(record["num_frames"])
    boundary_frame = max(1, min(int(boundary_frame), num_frames - 1))
    record["boundary_frame"] = boundary_frame
    record["segments"][0]["start_frame"] = 0
    record["segments"][0]["end_frame"] = boundary_frame
    record["segments"][1]["start_frame"] = boundary_frame
    record["segments"][1]["end_frame"] = num_frames
    record["review_status"] = "reviewed"
    record["confidence"] = 1.0
    old_flags = [flag for flag in record.get("flags", []) if not str(flag).startswith("needs_review")]
    record["flags"] = sorted(set(old_flags + ["human_reviewed"]))
    record["human_review"] = {
        "boundary_frame": boundary_frame,
        "boundary_time_s": boundary_frame / float(record["fps"]),
        "note": note,
    }


def _read_dual_view_jpeg(record: dict, frame_index: int) -> bytes:
    import cv2
    import h5py
    import numpy as np

    frame_index = max(0, min(int(frame_index), int(record["num_frames"]) - 1))
    with h5py.File(record["source_path"], "r") as handle:
        by_camera = handle.get("observations/images_by_camera")
        fallback = handle.get("observations/images")
        if by_camera is not None and "third_person" in by_camera:
            left = np.asarray(by_camera["third_person"][frame_index])
        elif fallback is not None:
            left = np.asarray(fallback[frame_index])
        else:
            raise KeyError(f"No images found in {record['source_path']}")

        if by_camera is not None and "wrist" in by_camera:
            right = np.asarray(by_camera["wrist"][frame_index])
        elif fallback is not None:
            right = np.asarray(fallback[frame_index])
        else:
            right = left

    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (right.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([left, right], axis=1)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode review frame")
    return encoded.tobytes()


PAGE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PI05 Subtask Review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1f2328; }
    header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #d8dee4; padding: 10px 16px; z-index: 2; }
    main { display: grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 52px); }
    aside { border-right: 1px solid #d8dee4; overflow: auto; max-height: calc(100vh - 52px); background: #fff; }
    button, input { font: inherit; }
    .item { display: block; width: 100%; padding: 10px 12px; border: 0; border-bottom: 1px solid #eef0f2; text-align: left; background: #fff; cursor: pointer; }
    .item.active { background: #ddf4ff; }
    .item.reviewed { opacity: .55; }
    .episode { font-weight: 650; font-size: 13px; }
    .meta { font-size: 12px; color: #57606a; margin-top: 3px; }
    section { padding: 18px; }
    video { width: min(100%, 980px); background: #111; display: block; }
    img.frame { width: min(100%, 980px); background: #111; display: block; image-rendering: auto; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
    .panel { background: #fff; border: 1px solid #d8dee4; padding: 12px; max-width: 980px; }
    .small { color: #57606a; font-size: 12px; }
    input[type=number] { width: 110px; }
    input[type=text] { min-width: 320px; flex: 1; }
    .ok { color: #1a7f37; }
    .bad { color: #cf222e; }
    pre { white-space: pre-wrap; overflow: auto; max-height: 180px; background: #f6f8fa; padding: 10px; }
  </style>
</head>
<body>
<header>
  <b>PI05 Subtask Review</b>
  <span id="counts" class="small"></span>
</header>
<main>
  <aside id="list"></aside>
  <section>
    <h3 id="title">Select an episode</h3>
    <img id="frameimg" class="frame" alt="review frame">
    <div class="panel">
      <div class="row">
        <label>frame <input id="frameSlider" type="range" min="0" value="0"></label>
        <button onclick="setBoundaryFromFrame()">Set Boundary To Shown Frame</button>
        <span id="frameLabel" class="small"></span>
      </div>
      <div class="row">
        <button onclick="useVlm()">Use VLM</button>
        <button onclick="useRobot()">Use Robot Signal</button>
        <label>boundary frame <input id="boundary" type="number" min="1"></label>
        <span id="time" class="small"></span>
      </div>
      <div class="row">
        <input id="note" type="text" placeholder="optional note">
        <button onclick="saveReview()">Accept / Save</button>
        <span id="status" class="small"></span>
      </div>
      <div id="details" class="small"></div>
      <pre id="raw"></pre>
    </div>
  </section>
</main>
<script>
let state = null;
let current = null;

async function loadState() {
  state = await (await fetch('/api/state')).json();
  renderList();
  if (state.queue.length) selectEpisode(state.queue[0]);
}

function record(id) { return state.records.find(r => r.episode_id === id); }

function renderList() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  const reviewed = state.records.filter(r => r.review_status === 'reviewed').length;
  document.getElementById('counts').textContent = `  total ${state.records.length}, queue ${state.queue.length}, reviewed ${reviewed}`;
  for (const id of state.queue) {
    const r = record(id);
    const b = document.createElement('button');
    b.className = 'item' + (current === id ? ' active' : '') + (r.review_status === 'reviewed' ? ' reviewed' : '');
    b.onclick = () => selectEpisode(id);
    b.innerHTML = `<div class="episode">${id}</div><div class="meta">${r.flags.join(', ') || r.review_status} | boundary ${r.boundary_frame}</div>`;
    list.appendChild(b);
  }
}

function selectEpisode(id) {
  current = id;
  const r = record(id);
  renderList();
  document.getElementById('title').textContent = id;
  document.getElementById('boundary').value = r.boundary_frame;
  const slider = document.getElementById('frameSlider');
  slider.max = Math.max(0, r.num_frames - 1);
  slider.value = r.boundary_frame;
  updateTime();
  updateFrame();
  const vlm = r.vlm && r.vlm.boundary_frame;
  const robot = r.refinement && r.refinement.frame;
  document.getElementById('details').innerHTML =
    `fps=${r.fps}, frames=${r.num_frames}, current=${r.boundary_frame}, vlm=${vlm}, robot=${robot}, flags=<span class="bad">${r.flags.join(', ')}</span>`;
  document.getElementById('raw').textContent = JSON.stringify(r.vlm && r.vlm.raw_output, null, 2);
  document.getElementById('status').textContent = '';
}

function updateTime() {
  const r = record(current);
  const b = Number(document.getElementById('boundary').value);
  document.getElementById('time').textContent = `${(b / r.fps).toFixed(2)}s original time`;
}

document.getElementById('boundary').addEventListener('input', updateTime);
document.getElementById('frameSlider').addEventListener('input', updateFrame);

function updateFrame() {
  const r = record(current);
  const frame = Number(document.getElementById('frameSlider').value);
  document.getElementById('frameLabel').textContent = `${frame}/${r.num_frames - 1}  ${(frame / r.fps).toFixed(2)}s`;
  document.getElementById('frameimg').src = `/api/frame?episode_id=${encodeURIComponent(current)}&frame=${frame}&t=${Date.now()}`;
}

function setBoundaryFromFrame() {
  const r = record(current);
  const frame = Number(document.getElementById('frameSlider').value);
  document.getElementById('boundary').value = Math.max(1, Math.min(r.num_frames - 1, frame));
  updateTime();
}

function useVlm() {
  const r = record(current);
  if (r.vlm && r.vlm.boundary_frame) document.getElementById('boundary').value = r.vlm.boundary_frame;
  document.getElementById('frameSlider').value = document.getElementById('boundary').value;
  updateTime();
  updateFrame();
}

function useRobot() {
  const r = record(current);
  if (r.refinement && r.refinement.frame) document.getElementById('boundary').value = r.refinement.frame;
  document.getElementById('frameSlider').value = document.getElementById('boundary').value;
  updateTime();
  updateFrame();
}

async function saveReview() {
  const boundary = Number(document.getElementById('boundary').value);
  const note = document.getElementById('note').value;
  const res = await fetch('/api/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({episode_id: current, boundary_frame: boundary, note})
  });
  const payload = await res.json();
  document.getElementById('status').innerHTML = res.ok ? `<span class="ok">${payload.message}</span>` : `<span class="bad">${payload.error}</span>`;
  await loadState();
}

loadState();
</script>
</body>
</html>
"""


def _make_handler(base_dir: Path, input_path: Path, output_path: Path, needs_review_only: bool):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _load_payload(self) -> dict:
            path = output_path if output_path.exists() else input_path
            with open(path) as handle:
                return json.load(handle)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                body = PAGE.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/state":
                payload = self._load_payload()
                records = payload["episodes"]
                if needs_review_only:
                    queue = [record["episode_id"] for record in records if _needs_review(record)]
                else:
                    queue = [record["episode_id"] for record in records]
                self._send_json({"records": records, "queue": queue})
                return

            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                episode_id = query.get("episode_id", [""])[0]
                frame = int(query.get("frame", ["0"])[0])
                payload = self._load_payload()
                for record in payload["episodes"]:
                    if record["episode_id"] == episode_id:
                        try:
                            body = _read_dual_view_jpeg(record, frame)
                        except Exception as exc:
                            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                            return
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                self._send_json({"error": f"unknown episode_id: {episode_id}"}, HTTPStatus.NOT_FOUND)
                return

            rel = unquote(parsed.path.lstrip("/"))
            target = (base_dir / rel).resolve()
            if not str(target).startswith(str(base_dir.resolve())) or not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(target)

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            rel = unquote(parsed.path.lstrip("/"))
            target = (base_dir / rel).resolve()
            if not str(target).startswith(str(base_dir.resolve())) or not target.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(target, head_only=True)

        def _serve_file(self, target: Path, head_only: bool = False) -> None:
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            size = target.stat().st_size
            range_header = self.headers.get("Range")
            start = 0
            end = size - 1
            status = HTTPStatus.OK
            if range_header and range_header.startswith("bytes="):
                raw = range_header.removeprefix("bytes=").split(",", 1)[0]
                raw_start, _, raw_end = raw.partition("-")
                if raw_start:
                    start = int(raw_start)
                if raw_end:
                    end = int(raw_end)
                end = min(end, size - 1)
                if 0 <= start <= end:
                    status = HTTPStatus.PARTIAL_CONTENT
                else:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head_only:
                return
            with open(target, "rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(length))

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            episode_id = str(request.get("episode_id", ""))
            boundary_frame = int(request.get("boundary_frame", 0))
            note = str(request.get("note", ""))
            payload = self._load_payload()
            for record in payload["episodes"]:
                if record["episode_id"] == episode_id:
                    _update_record_boundary(record, boundary_frame, note)
                    _atomic_write_json(output_path, payload)
                    self._send_json({"message": f"saved {episode_id} -> {output_path}"})
                    return
            self._send_json({"error": f"unknown episode_id: {episode_id}"}, HTTPStatus.NOT_FOUND)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[review] {self.address_string()} - {fmt % args}")

    return Handler


def main() -> None:
    args = tyro.cli(Args)
    input_path = Path(args.annotations_path).expanduser().resolve()
    if args.output_path is None:
        output_path = input_path.with_name("annotations_reviewed.json")
    else:
        output_path = Path(args.output_path).expanduser().resolve()
    base_dir = input_path.parent
    handler = _make_handler(base_dir, input_path, output_path, args.needs_review_only)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Review server: http://{args.host}:{args.port}/")
    print(f"Input sidecar: {input_path}")
    print(f"Reviewed sidecar: {output_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
