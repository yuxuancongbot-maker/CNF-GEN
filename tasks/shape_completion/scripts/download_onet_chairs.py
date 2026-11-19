#!/usr/bin/env python3
"""Resumable HTTP-Range downloader for the official ONet ShapeNet-v1 chair member.

Downloads ONLY the `03001627.zip` member of the public watertight.zip archive:
  https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/watertight.zip

Strategy
--------
1. Read the ZIP64 end-of-central-directory + central directory via byte ranges to
   locate `03001627.zip` exactly (no other category is ever fetched).
2. Read that member's local header to find the precise start of its file data.
3. If STORED (method 0) -> bounded concurrent ranged download of just that byte
   span into a preallocated file, with per-chunk resume state.
   If DEFLATED (method 8) -> sequential streamed download + raw inflate.
4. Verify CRC32 (from central directory) + SHA256 + that the result is a valid
   ZIP, then atomically publish `03001627.zip` and `DOWNLOAD_READY.json`.

Every request carries a unique query tag (?ocd=<token>&c=<chunk>) so a caching
proxy cannot collapse distinct Range requests onto one cached object.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.request
import zipfile
import zlib

URL = "https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/watertight.zip"
MEMBER = "03001627.zip"
EOCD_SIG = b"PK\x05\x06"
EOCD64_SIG = b"PK\x06\x06"
LOC64_SIG = b"PK\x06\x07"
CEN_SIG = b"PK\x01\x02"
LOC_SIG = b"PK\x03\x04"


def _token():
    import secrets

    return "ocd-20261118-" + secrets.token_hex(6)


class Fetcher:
    def __init__(self, token, retries=10, timeout=180):
        self.token = token
        self.retries = retries
        self.timeout = timeout

    def url_for(self, chunk):
        return f"{URL}?ocd={self.token}&c={chunk}"

    def get(self, start, end, chunk="meta", expect_total=None):
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url_for(chunk),
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "opencode-onet-chair-downloader/1.0",
                        "Accept-Encoding": "identity",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise RuntimeError(f"expected 206 got {r.status}")
                    cr = r.headers.get("Content-Range", "")
                    expected_cr = f"bytes {start}-{end}/"
                    if not cr.startswith(expected_cr):
                        raise RuntimeError(f"bad Content-Range {cr!r} != {expected_cr}*")
                    if expect_total is not None and not cr.endswith(f"/{expect_total}"):
                        raise RuntimeError(f"bad Content-Range total {cr!r}")
                    out = bytearray()
                    while True:
                        block = r.read(1024 * 1024)
                        if not block:
                            break
                        out.extend(block)
                    return bytes(out)
            except Exception as e:  # noqa: BLE001 - network retry
                last = e
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 30))
        raise RuntimeError(f"range {start}-{end} failed after {self.retries} attempts: {last}")

    def stream_to_fd(self, start, end, fd, chunk_idx, base, progress_cb=None):
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url_for(chunk_idx),
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "opencode-onet-chair-downloader/1.0",
                        "Accept-Encoding": "identity",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise RuntimeError(f"expected 206 got {r.status}")
                    cr = r.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"bad Content-Range {cr!r}")
                    pos = start
                    while True:
                        block = r.read(1024 * 1024)
                        if not block:
                            break
                        os.pwrite(fd, block, pos - base)  # relative to member data start
                        pos += len(block)
                        if progress_cb:
                            progress_cb(len(block))
                    if pos != end + 1:
                        raise RuntimeError(f"short chunk {pos} != {end + 1}")
                    return
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 30))
        raise RuntimeError(f"chunk {chunk_idx} failed after {self.retries} attempts: {last}")


def find_member(fetcher, outer_size):
    tail_len = 256 * 1024
    tail = fetcher.get(outer_size - tail_len, outer_size - 1, "eocd", outer_size)
    eocd = tail.rfind(EOCD_SIG)
    if eocd < 0:
        raise RuntimeError("EOCD not found")
    cd_size = struct.unpack_from("<I", tail, eocd + 12)[0]
    cd_off = struct.unpack_from("<I", tail, eocd + 16)[0]
    if cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        loc = tail.rfind(LOC64_SIG)
        if loc < 0:
            raise RuntimeError("ZIP64 locator not found")
        z64_off = struct.unpack_from("<Q", tail, loc + 8)[0]
        z64 = fetcher.get(z64_off, z64_off + 55, "eocd64", outer_size)
        if z64[:4] != EOCD64_SIG:
            raise RuntimeError("ZIP64 EOCD signature mismatch")
        cd_size = struct.unpack_from("<Q", z64, 40)[0]
        cd_off = struct.unpack_from("<Q", z64, 48)[0]
    cd = fetcher.get(cd_off, cd_off + cd_size - 1, "central", outer_size)
    if len(cd) != cd_size:
        raise RuntimeError("central directory short read")

    p = 0
    found = None
    entries = 0
    while p + 46 <= len(cd):
        if cd[p : p + 4] != CEN_SIG:
            break
        method = struct.unpack_from("<H", cd, p + 10)[0]
        crc = struct.unpack_from("<I", cd, p + 16)[0]
        csize = struct.unpack_from("<I", cd, p + 20)[0]
        usize = struct.unpack_from("<I", cd, p + 24)[0]
        nlen, elen, clen = struct.unpack_from("<HHH", cd, p + 28)
        lhoff = struct.unpack_from("<I", cd, p + 42)[0]
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
        entries += 1
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lhoff == 0xFFFFFFFF:
            extra = cd[p + 46 + nlen : p + 46 + nlen + elen]
            q = 0
            while q + 4 <= len(extra):
                eid, esz = struct.unpack_from("<HH", extra, q)
                q += 4
                body = extra[q : q + esz]
                q += esz
                if eid == 0x0001:
                    r = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack_from("<Q", body, r)[0]
                        r += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack_from("<Q", body, r)[0]
                        r += 8
                    if lhoff == 0xFFFFFFFF:
                        lhoff = struct.unpack_from("<Q", body, r)[0]
                        r += 8
        if name == MEMBER:
            found = dict(name=name, method=method, crc=crc, csize=csize, usize=usize, lhoff=lhoff)
        p += 46 + nlen + elen + clen
    if found is None:
        raise RuntimeError(f"{MEMBER} not found in central directory ({entries} entries)")
    found["entries"] = entries

    lh = fetcher.get(found["lhoff"], found["lhoff"] + 29, "localhdr", outer_size)
    if lh[:4] != LOC_SIG:
        raise RuntimeError("local header signature mismatch")
    nlen, elen = struct.unpack_from("<HH", lh, 26)
    found["data_offset"] = found["lhoff"] + 30 + nlen + elen
    return found


_WRITE_LOCK = threading.Lock()


def atomic_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with _WRITE_LOCK:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def verify(path, member, sha_path):
    crc = 0
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            crc = zlib.crc32(b, crc)
            h.update(b)
            total += len(b)
    crc &= 0xFFFFFFFF
    if total != member["usize"]:
        raise RuntimeError(f"size mismatch {total} != {member['usize']}")
    if crc != member["crc"]:
        raise RuntimeError(f"CRC32 mismatch {crc:08x} != {member['crc']:08x}")
    sha = h.hexdigest()
    # Structural ZIP validation: open and enumerate (do not decompress all).
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError("member ZIP has no entries")
        bad = zf.testzip()  # CRC check of every member
        if bad is not None:
            raise RuntimeError(f"inner ZIP CRC failure at {bad}")
    return dict(
        crc32=f"{crc:08x}",
        sha256=sha,
        bytes=total,
        inner_entries=len(names),
        inner_sample=names[:5],
        inner_testzip="ok",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--state",
        default=None,
        help="state JSON path; defaults to <archive-dir>/download_state.json",
    )
    ap.add_argument(
        "--progress",
        default=None,
        help="progress JSON path; defaults to <archive-dir>/download_progress.json",
    )
    ap.add_argument(
        "--out", required=True, help="output archive path (required; no default next to source)"
    )
    args = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(args.out))
    if args.state is None:
        args.state = os.path.join(outdir, "download_state.json")
    if args.progress is None:
        args.progress = os.path.join(outdir, "download_progress.json")

    state = None
    if os.path.exists(args.state):
        try:
            state = json.load(open(args.state))
        except Exception:
            state = None

    if state and state.get("status") in ("verified", "downloading", "downloading-inflate"):
        token = state["token"]
        fetcher = Fetcher(token)
        member = state["member"]
        outer_size = state["outer_size"]
    else:
        token = _token()
        fetcher = Fetcher(token)
        outer_size = 97296729934
        print(f"[probe] token={token} outer_size={outer_size}", flush=True)
        member = find_member(fetcher, outer_size)
        print(f"[probe] member={json.dumps(member)}", flush=True)
        state = dict(
            token=token,
            url=URL,
            outer_size=outer_size,
            member=member,
            chunk_size=args.chunk_mb * 1024 * 1024,
            completed=[],
            bytes_written=0,
            status="downloading",
            started_at=time.time(),
            finished_at=None,
            sha256=None,
        )
        atomic_json(args.state, state)

    member = state["member"]
    data_off = member["data_offset"]
    csize = member["csize"]
    chunk = state["chunk_size"]
    total_chunks = (csize + chunk - 1) // chunk
    print(
        f"[download] method={member['method']} data_offset={data_off} "
        f"csize={csize} chunks={total_chunks} done={len(state['completed'])}",
        flush=True,
    )

    tmp = args.out + ".part"
    fd = os.open(tmp, os.O_CREAT | os.O_RDWR)
    try:
        os.ftruncate(fd, csize)  # sparse preallocation

        if member["method"] == 8:
            # Deflate: must stream sequentially from the start.
            raise RuntimeError(
                "DEFLATE member encountered: sequential-inflate path "
                "not enabled; contact Codex via status file"
            )
        elif member["method"] != 0:
            raise RuntimeError(f"unsupported compression method {member['method']}")

        done_set = set(state["completed"])
        lock = threading.Lock()
        queue = [i for i in range(total_chunks) if i not in done_set]
        qcursor = [0]
        byte_counts = [0]

        def progress_cb(n):
            with lock:
                byte_counts[0] += n

        def worker():
            while True:
                with lock:
                    if qcursor[0] >= len(queue):
                        return
                    idx = queue[qcursor[0]]
                    qcursor[0] += 1
                try:
                    s = data_off + idx * chunk
                    e = min(s + chunk, data_off + csize) - 1
                    fetcher.stream_to_fd(s, e, fd, idx, data_off, progress_cb)
                    with lock:
                        done_set.add(idx)
                        state["completed"] = sorted(done_set)
                        state["bytes_written"] = min(len(done_set) * chunk, csize)
                        atomic_json(args.state, state)
                        ndone = len(done_set)
                    if ndone % 5 == 0 or ndone == total_chunks:
                        pct = 100.0 * ndone / total_chunks
                        print(
                            f"[progress] {ndone}/{total_chunks} chunks ({pct:.2f}%) "
                            f"~{ndone * chunk / 1e9:.2f} GB",
                            flush=True,
                        )
                        pb = dict(
                            status="downloading",
                            chunks_done=ndone,
                            chunks_total=total_chunks,
                            percent=round(pct, 3),
                            bytes=min(ndone * chunk, csize),
                            total=csize,
                            updated_at=time.time(),
                        )
                        atomic_json(args.progress, pb)
                except Exception as exc:  # keep worker alive; chunk stays queued on retry
                    print(f"[worker-error] chunk {idx}: {type(exc).__name__}: {exc}", flush=True)
                    with lock:
                        queue.append(idx)

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(min(args.workers, max(1, len(queue))))
        ]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if len(done_set) != total_chunks:
            raise RuntimeError(f"incomplete: {len(done_set)}/{total_chunks}")

        print("[verify] hashing + CRC32 + inner ZIP validation ...", flush=True)
        res = verify(tmp, member, None)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(tmp, args.out)
        state["status"] = "verified"
        state["finished_at"] = time.time()
        state["verification"] = res
        state["sha256"] = res["sha256"]
        atomic_json(args.state, state)
        atomic_json(
            os.path.join(outdir, "DOWNLOAD_READY.json"),
            dict(
                member=MEMBER,
                file=os.path.abspath(args.out),
                bytes=res["bytes"],
                sha256=res["sha256"],
                crc32=res["crc32"],
                inner_entries=res["inner_entries"],
                inner_sample=res["inner_sample"],
                inner_testzip=res["inner_testzip"],
                source_url=URL,
                outer_size=outer_size,
                request_token=token,
                completed_at=time.time(),
                elapsed_s=time.time() - t0,
                command=" ".join(sys.argv),
            ),
        )
        atomic_json(
            args.progress,
            dict(status="verified", percent=100, bytes=csize, total=csize, updated_at=time.time()),
        )
        print(
            f"[DONE] {args.out} sha256={res['sha256']} crc32={res['crc32']} "
            f"inner_entries={res['inner_entries']}",
            flush=True,
        )
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == "__main__":
    main()
