"""Download ONLY the official NYUv2 labeled MAT using resumable, verified HTTP ranges.

Default: four workers, 32 MiB chunks. --output is the assembled .partial FILE,
not a directory. This uses only Python's standard library. Afterwards run the
separate prepare_nyuv2.py HDF5/content checks before using the MAT for training.

Example:
  python download_nyuv2_ranges.py --output /path/to/cache/nyu_depth_v2_labeled.mat.partial
  python download_nyuv2_ranges.py --self-test
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import urllib.request


# Same official source and exact HTTPS HEAD size verified by prepare_nyuv2.py.
SOURCE_URL = 'https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
SOURCE_BYTES = 2_972_037_809
CHUNK_BYTES = 32 * 1024 * 1024
READ_BYTES = 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 20
PRINT_LOCK = threading.Lock()


class Cancelled(RuntimeError):
    pass


class ProtocolError(RuntimeError):
    """Not retryable: server did not honor the exact requested range."""


@dataclass(frozen=True)
class Spec:
    url: str = SOURCE_URL
    size: int = SOURCE_BYTES
    chunk_size: int = CHUNK_BYTES

    def metadata(self):
        return dict(url=self.url, bytes=self.size, chunk_bytes=self.chunk_size,
                    integrity='SHA256 is locally observed; no publisher-provided MAT checksum is available')

    def ranges(self):
        return [(start, min(start + self.chunk_size, self.size) - 1)
                for start in range(0, self.size, self.chunk_size)]


def say(message):
    with PRINT_LOCK:
        print(message, flush=True)


def check_cancel(stop):
    if stop.is_set():
        raise Cancelled('Download cancelled; verified complete chunks remain reusable')


def digest_file(path, stop=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(READ_BYTES), b''):
            if stop is not None:
                check_cancel(stop)
            digest.update(data)
    return digest.hexdigest()


@contextmanager
def own_temporary(directory, prefix):
    fd, name = tempfile.mkstemp(dir=directory, prefix=prefix, suffix='.incomplete')
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def publish_no_replace(temporary, target):
    """Same-filesystem hard link creates the destination atomically without overwriting."""
    os.link(temporary, target)


def ensure_json(path, value):
    """Create metadata once, or verify an existing identical value without replacing it."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Conflicting existing metadata: {path}; choose another --output')
        return
    with own_temporary(path.parent, path.name + '.') as temporary:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        publish_no_replace(temporary, path)


def chunk_prefix(start, end):
    return f'{start:012d}-{end:012d}'


def reusable_chunk(cache, start, end, stop):
    """Only full SHA-named chunks are reusable; validate both size and their recorded hash."""
    prefix = chunk_prefix(start, end)
    candidates = list(cache.glob(prefix + '.*.chunk'))
    if len(candidates) > 1:
        raise ValueError(f'Conflicting completed chunks for bytes {start}-{end}: {candidates}')
    if not candidates:
        return None
    path = candidates[0]
    match = re.fullmatch(re.escape(prefix) + r'\.([0-9a-f]{64})\.chunk', path.name)
    if not match or path.stat().st_size != end - start + 1:
        raise ValueError(f'Invalid completed chunk metadata/size: {path}')
    actual = digest_file(path, stop)
    if actual != match[1]:
        raise ValueError(f'Completed chunk SHA256 mismatch: {path}; preserved for manual inspection')
    return path, actual


def stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def seed_from_prefix(prefix, output, cache, spec, stop):
    """Read a caller-confirmed, STOPPED same-URL curl prefix without modifying it.

    Only complete fixed-size chunks are staged. Verify source stability and all
    existing-cache conflicts before publishing any new chunks. This does not
    prove the prefix came from the declared server; caller confirmation is required.
    """
    prefix = Path(prefix).resolve()
    if prefix == output.resolve():
        raise ValueError('--seed-prefix must not be the --output file; choose a new output')
    if not prefix.is_file():
        raise ValueError(f'Seed prefix is not a regular file: {prefix}')
    initial_stat = prefix.stat()
    if initial_stat.st_size < spec.chunk_size or initial_stat.st_size > spec.size:
        raise ValueError(f'Seed size must be between one full chunk and {spec.size} bytes')
    source_hash = hashlib.sha256()
    staged = []
    with tempfile.TemporaryDirectory(prefix='.seed-stage-', dir=cache) as directory:
        stage_root = Path(directory)
        with prefix.open('rb') as source:
            if stat_identity(os.fstat(source.fileno())) != stat_identity(initial_stat):
                raise ValueError('Seed source changed before reading; stop curl before seeding')
            offset = 0
            while offset < initial_stat.st_size:
                check_cancel(stop)
                count = min(spec.chunk_size, initial_stat.st_size - offset)
                is_full_chunk = count == spec.chunk_size
                staged_path = stage_root / f'{offset:012d}.chunk'
                chunk_hash = hashlib.sha256()
                target_context = staged_path.open('wb') if is_full_chunk else nullcontext(None)
                with target_context as target:
                    remaining = count
                    while remaining:
                        check_cancel(stop)
                        data = source.read(min(READ_BYTES, remaining))
                        if not data:
                            raise ValueError('Seed source was truncated while reading; no chunks imported')
                        source_hash.update(data)
                        chunk_hash.update(data)
                        if target is not None:
                            target.write(data)
                        remaining -= len(data)
                    if target is not None:
                        target.flush()
                        os.fsync(target.fileno())
                if is_full_chunk:
                    staged.append((offset, offset + count - 1, staged_path, chunk_hash.hexdigest()))
                offset += count
            if source.read(1) or stat_identity(os.fstat(source.fileno())) != stat_identity(initial_stat):
                raise ValueError('Seed source changed while reading; no chunks imported; stop curl first')
        if stat_identity(prefix.stat()) != stat_identity(initial_stat):
            raise ValueError('Seed source path changed while reading; no chunks imported')
        missing = []
        for start, end, staged_path, checksum in staged:
            reused = reusable_chunk(cache, start, end, stop)
            if reused:
                if reused[1] != checksum:
                    raise ValueError(f'Seed prefix conflicts with cached bytes {start}-{end}; no chunks imported')
            else:
                missing.append((start, end, staged_path, checksum))
        provenance = dict(
            source_path=str(prefix), source_bytes=initial_stat.st_size, source_sha256=source_hash.hexdigest(),
            declared_source_url=spec.url,
            source_trust='Caller explicitly confirmed same official URL and stopped transfer; URL provenance is not independently proven',
            full_chunk_bytes=sum(end - start + 1 for start, end, _, _ in staged),
            ignored_trailing_bytes=initial_stat.st_size % spec.chunk_size,
            eligible_chunks=[dict(start=start, end=end, sha256=checksum) for start, end, _, checksum in staged],
            validation='Source stat unchanged during reading; local hashes only; final HDF5/content validation still required',
        )
        provenance_dir = cache / 'seed_sources'
        provenance_dir.mkdir(exist_ok=True)
        provenance_id = hashlib.sha256((str(prefix) + '\n' + provenance['source_sha256']).encode()).hexdigest()
        # Record potential seed provenance before publishing, so a crash cannot
        # leave seeded chunks without any source record. Not all eligible chunks
        # are necessarily newly imported (some may already exist identically).
        ensure_json(provenance_dir / f'{provenance_id}.json', provenance)
        for start, end, staged_path, checksum in missing:
            check_cancel(stop)
            destination = cache / f'{chunk_prefix(start, end)}.{checksum}.chunk'
            publish_no_replace(staged_path, destination)
        say(f'Seeded {len(missing)} new complete chunks from {prefix}; '
            f'{len(staged) - len(missing)} already matched; '
            f'{provenance["ignored_trailing_bytes"]} trailing bytes ignored; '
            f'source SHA256 {provenance["source_sha256"]}')
        return provenance


def fetch_attempt(spec, cache, start, end, stop, opener):
    check_cancel(stop)
    expected_length = end - start + 1
    request = urllib.request.Request(spec.url, headers={
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
        'User-Agent': 'AI-Class-NYUv2-Range-Download/1.0',
    })
    with own_temporary(cache, chunk_prefix(start, end) + '.') as temporary:
        digest = hashlib.sha256()
        received = 0
        with opener(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            if response.status != 206:
                # In particular, never read a 200 response: it might be the entire 3 GB file.
                raise ProtocolError(f'Expected HTTP 206 for a range, got {response.status}; body NOT read')
            if response.geturl() != spec.url:
                raise ProtocolError(f'Unexpected redirect destination: {response.geturl()}')
            expected_range = f'bytes {start}-{end}/{spec.size}'
            if response.headers.get('Content-Range') != expected_range:
                raise ProtocolError(f'Expected Content-Range {expected_range!r}, got {response.headers.get("Content-Range")!r}')
            encoding = response.headers.get('Content-Encoding', 'identity').lower()
            if encoding not in ('', 'identity'):
                raise ProtocolError(f'Unexpected encoded range body: {encoding}')
            length = response.headers.get('Content-Length')
            if length is not None and (not length.isdigit() or int(length) != expected_length):
                raise ProtocolError(f'Unexpected Content-Length {length}, expected {expected_length}')
            with temporary.open('wb') as stream:
                # HTTPResponse.read1 returns after one socket read, making cancellation
                # responsive even when a slow peer trickles less than a full MiB.
                read_once = getattr(response, 'read1', response.read)
                while True:
                    check_cancel(stop)
                    # One byte beyond expected is sufficient to detect an oversized response.
                    data = read_once(min(READ_BYTES, expected_length - received + 1))
                    if not data:
                        break
                    received += len(data)
                    if received > expected_length:
                        raise ProtocolError('Range body exceeds requested size')
                    digest.update(data)
                    stream.write(data)
                if received != expected_length:
                    raise OSError(f'Truncated range {start}-{end}: {received}/{expected_length} bytes')
                stream.flush()
                os.fsync(stream.fileno())
        check_cancel(stop)
        actual = digest.hexdigest()
        target = cache / f'{chunk_prefix(start, end)}.{actual}.chunk'
        publish_no_replace(temporary, target)
        return target, actual


def fetch_chunk(spec, cache, start, end, stop, retries, opener):
    reused = reusable_chunk(cache, start, end, stop)
    if reused:
        return start, end, *reused, True
    for attempt in range(1, retries + 1):
        try:
            path, actual = fetch_attempt(spec, cache, start, end, stop, opener)
            return start, end, path, actual, False
        except (Cancelled, ProtocolError):
            raise
        except Exception as error:
            check_cancel(stop)
            if attempt == retries:
                raise RuntimeError(f'Range {start}-{end} failed after {retries} attempts') from error
            delay = min(2 ** (attempt - 1), 4)
            say(f'Retry range {start}-{end}, attempt {attempt + 1}/{retries}: {type(error).__name__}: {error}')
            if stop.wait(delay):
                raise Cancelled('Cancelled while waiting to retry')
    raise AssertionError('Unreachable')


def assemble(output, chunks, spec, stop):
    """Assemble only complete verified chunks, with no overwrite of an existing output."""
    if output.exists():
        if output.stat().st_size != spec.size:
            raise ValueError(f'Existing output has a conflicting size: {output}; preserved')
        # The expected hash is computed from verified chunks, not trusted solely from a sidecar.
        candidate_hash = digest_file(output, stop)
    else:
        candidate_hash = None
    with own_temporary(output.parent, output.name + '.assembling.') as temporary:
        digest = hashlib.sha256()
        received = 0
        with temporary.open('wb') as assembled:
            for start, end, path, expected_hash, _ in sorted(chunks):
                if start != received:
                    raise ValueError('Chunks are not contiguous')
                chunk_digest = hashlib.sha256()
                copied = 0
                with path.open('rb') as source:
                    for data in iter(lambda: source.read(READ_BYTES), b''):
                        check_cancel(stop)
                        digest.update(data)
                        chunk_digest.update(data)
                        if candidate_hash is None:
                            assembled.write(data)
                        copied += len(data)
                if copied != end - start + 1 or chunk_digest.hexdigest() != expected_hash:
                    raise ValueError(f'Chunk changed during assembly: {path}')
                received += copied
            assembled.flush()
            os.fsync(assembled.fileno())
        if received != spec.size:
            raise ValueError(f'Assembly has {received} bytes, expected {spec.size}')
        actual = digest.hexdigest()
        if candidate_hash is not None:
            if candidate_hash != actual:
                raise ValueError(f'Existing output conflicts with verified chunks: {output}; preserved')
        else:
            check_cancel(stop)
            publish_no_replace(temporary, output)
    cache = output.with_name(output.name + '.chunks')
    seed_sources = [json.loads(path.read_text()) for path in sorted((cache / 'seed_sources').glob('*.json'))]
    manifest = dict(source=spec.metadata(), output=str(output.resolve()), bytes=received, sha256=actual,
                    chunk_count=len(chunks),
                    validation='Byte counts/local SHA256 verified; see seed_sources for any caller-confirmed curl prefix. HDF5/content validation still required',
                    seed_sources=seed_sources,
                    chunks=[dict(start=start, end=end, sha256=checksum, file=path.name)
                            for start, end, path, checksum, _ in sorted(chunks)])
    ensure_json(output.with_name(output.name + '.sha256.json'), manifest)
    return manifest


def run(output, workers=4, retries=3, spec=Spec(), opener=None, verbose=True,
        seed_prefix=None, confirm_seed_official_source=False):
    if opener is None:
        opener = urllib.request.urlopen
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (not output.is_file() or output.stat().st_size != spec.size):
        raise ValueError(f'--output already exists with conflicting type/size: {output}; preserved')
    cache = output.with_name(output.name + '.chunks')
    cache.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    with (cache / '.download.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another downloader holds this output cache lock')
        ensure_json(cache / 'source.json', spec.metadata())
        if seed_prefix is not None:
            if not confirm_seed_official_source:
                raise ValueError('Seeding requires explicit --confirm-seed-official-source confirmation')
            seed_from_prefix(seed_prefix, output, cache, spec, stop)
        chunks, completed = [], 0
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='nyuv2-range')
        futures = [executor.submit(fetch_chunk, spec, cache, start, end, stop, retries, opener)
                   for start, end in spec.ranges()]
        try:
            for future in as_completed(futures):
                chunk = future.result()
                chunks.append(chunk)
                completed += chunk[1] - chunk[0] + 1
                if verbose:
                    elapsed = max(time.monotonic() - started, 0.001)
                    say(f'Complete {len(chunks)}/{len(futures)} chunks; {completed / 1e6:.1f}/{spec.size / 1e6:.1f} MB; '
                        f'{completed / elapsed / 1e6:.2f} MB/s including cache; '
                        f'{"reused" if chunk[-1] else "downloaded"} bytes {chunk[0]}-{chunk[1]}')
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            # Workers check cancellation per read; outstanding network calls use a 20 s timeout.
            executor.shutdown(wait=True, cancel_futures=True)
        return assemble(output, chunks, spec, stop)


def self_test():
    """In-memory HTTP mock: no network and no real NYUv2 download."""
    payload = bytes(range(251)) * 103
    spec = Spec(size=len(payload), chunk_size=4096)
    calls = []
    calls_lock = threading.Lock()

    class Response(io.BytesIO):
        def __init__(self, data, start, end, status=206, bad_range=False):
            super().__init__(data)
            self.status = status
            self.headers = {'Content-Length': str(len(data)),
                            'Content-Range': 'bad' if bad_range else f'bytes {start}-{end}/{spec.size}'}
            self.read_calls = 0

        def geturl(self):
            return spec.url

        def read(self, size=-1):
            self.read_calls += 1
            return super().read(size)

    def opener(request, timeout):
        assert timeout == NETWORK_TIMEOUT_SECONDS
        assert request.full_url == SOURCE_URL
        start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        with calls_lock:
            calls.append((start, end))
        return Response(payload[start:end + 1], start, end)

    with tempfile.TemporaryDirectory(prefix='nyuv2-range-selftest-') as directory:
        output = Path(directory) / 'test.mat.partial'
        manifest = run(output, spec=spec, opener=opener, verbose=False)
        assert output.read_bytes() == payload
        assert manifest['sha256'] == hashlib.sha256(payload).hexdigest()
        count = len(calls)
        assert count == len(spec.ranges())
        run(output, spec=spec, opener=opener, verbose=False)
        assert len(calls) == count, 'Second run must reuse all SHA-verified chunks'
        # HTTP 200 MUST be rejected before even reading its body.
        stop = threading.Event()
        cache = output.with_name(output.name + '.chunks')
        response = Response(payload, 0, 4095, status=200)
        try:
            fetch_attempt(spec, cache, 0, 4095, stop, lambda *a, **k: response)
            raise AssertionError('HTTP 200 unexpectedly accepted')
        except ProtocolError:
            assert response.read_calls == 0
        response = Response(payload[:4096], 0, 4095, bad_range=True)
        try:
            fetch_attempt(spec, cache, 0, 4095, stop, lambda *a, **k: response)
            raise AssertionError('Wrong Content-Range unexpectedly accepted')
        except ProtocolError:
            assert response.read_calls == 0
        stop.set()
        try:
            fetch_attempt(spec, cache, 0, 4095, stop, opener)
            raise AssertionError('Cancelled worker unexpectedly started a transfer')
        except Cancelled:
            assert len(calls) == count
        stop.clear()
        failures = []

        def always_fails(request, timeout):
            failures.append(1)
            raise OSError('mock disconnection')

        fresh_cache = Path(directory) / 'retry-cache'
        fresh_cache.mkdir()
        try:
            fetch_chunk(spec, fresh_cache, 0, 4095, stop, 2, always_fails)
            raise AssertionError('Persistent network failure unexpectedly succeeded')
        except RuntimeError:
            assert len(failures) == 2, 'Retries must be finite and match the configured attempt count'
        assert not list(fresh_cache.glob('*.incomplete'))
        # Corrupted completed chunks must not silently be redownloaded or reused.
        first_chunk = sorted(cache.glob('*.chunk'))[0]
        with first_chunk.open('r+b') as stream:
            stream.write(b'\xff')
        try:
            run(output, spec=spec, opener=opener, verbose=False)
            raise AssertionError('Corrupt cached chunk unexpectedly accepted')
        except ValueError:
            pass
        assert output.read_bytes() == payload, 'Existing assembled file must remain untouched'
        assert not list(cache.glob('*.incomplete'))
        # Prefix includes two full chunks plus a short tail: only the two full
        # chunks may be reused, while every remaining range must be requested.
        prefix = Path(directory) / 'stopped-curl.partial'
        prefix_data = payload[:2 * spec.chunk_size + 99]
        prefix.write_bytes(prefix_data)
        seeded_output = Path(directory) / 'seeded.mat.partial'
        before_calls = len(calls)
        seeded_manifest = run(seeded_output, spec=spec, opener=opener, verbose=False,
                              seed_prefix=prefix, confirm_seed_official_source=True)
        assert prefix.read_bytes() == prefix_data, 'Prefix source must never be modified'
        assert seeded_output.read_bytes() == payload
        assert len(calls) - before_calls == len(spec.ranges()) - 2
        provenance = seeded_manifest['seed_sources'][0]
        assert provenance['source_bytes'] == len(prefix_data)
        assert provenance['source_sha256'] == hashlib.sha256(prefix_data).hexdigest()
        assert provenance['ignored_trailing_bytes'] == 99
        assert len(provenance['eligible_chunks']) == 2
        conflict_prefix = Path(directory) / 'conflicting-curl.partial'
        conflict_prefix.write_bytes(b'\xff' + prefix_data[1:])
        before_calls = len(calls)
        try:
            run(seeded_output, spec=spec, opener=opener, verbose=False,
                seed_prefix=conflict_prefix, confirm_seed_official_source=True)
            raise AssertionError('Conflicting prefix unexpectedly accepted')
        except ValueError:
            assert len(calls) == before_calls, 'Conflict must fail before network transfers start'
        assert seeded_output.read_bytes() == payload
        try:
            run(seeded_output, spec=spec, opener=opener, verbose=False, seed_prefix=prefix)
            raise AssertionError('Seed source without confirmation unexpectedly accepted')
        except ValueError:
            pass
    say('PASS: exact parallel ranges, SHA/byte assembly, cache reuse, 200 rejection without body reads, '
        'Content-Range rejection, cancellation, finite retries, corruption detection, conflict preservation, '
        'read-only prefix import, full-chunk-only reuse, provenance, seed conflict/confirmation checks and temporary cleanup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Assembled .partial FILE; adjacent .chunks directory stores reusable verified blocks')
    parser.add_argument('--workers', type=int, default=4, help='1-8 concurrent ranges; default 4')
    parser.add_argument('--retries', type=int, default=3, help='1-6 total attempts per range; default 3')
    parser.add_argument('--seed-prefix', type=Path, help='Optional STOPPED curl prefix from the exact same official URL; source is never modified')
    parser.add_argument('--confirm-seed-official-source', action='store_true',
                        help='Explicitly confirm seed prefix came from the exact official source URL and its download has stopped')
    parser.add_argument('--self-test', action='store_true', help='Run offline in-memory HTTP tests only')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error('--output is required; it must not name an active curl output')
    if not 1 <= args.workers <= 8 or not 1 <= args.retries <= 6:
        parser.error('--workers must be 1..8 and --retries must be 1..6')
    if bool(args.seed_prefix) != args.confirm_seed_official_source:
        parser.error('--seed-prefix and --confirm-seed-official-source must be supplied together')
    try:
        result = run(args.output, args.workers, args.retries, seed_prefix=args.seed_prefix,
                     confirm_seed_official_source=args.confirm_seed_official_source)
    except KeyboardInterrupt:
        raise SystemExit('Cancelled. Completed verified chunks are retained; rerun the same command to resume.')
    print(json.dumps({key: result[key] for key in ('output', 'bytes', 'sha256', 'chunk_count', 'validation')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
