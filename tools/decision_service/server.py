"""Loopback service around the resident uzu typed-decision worker.

The service owns model processes and technical inference receipts only. It does
not own research state, action admission, experiment execution or release policy.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .telemetry import MemorySample, memory, power


def encoded(value, *, canonical=False):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=canonical, separators=(',', ':')).encode()


def digest(value, *, canonical=False):
    return hashlib.sha256(encoded(value, canonical=canonical)).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'invalid number: {value}')))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ServiceError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status

    def body(self):
        return {'code': self.code, 'message': str(self)}


def validate_response(request, response):
    """Validate the transport boundary even though the Rust core also validates it."""
    if (not isinstance(response, dict) or not isinstance(response.get('answers'), dict)
            or set(response['answers']) != set(request['questions'])):
        raise ServiceError('invalid_output', 'question identities differ', 502)
    for qid, question in request['questions'].items():
        kind = question['type']
        options = (list(question['criteria']) if kind == 'choice' else
                   [str(i) for i in range(len(question['criteria']))] if kind == 'score' else ['false', 'true'])
        answer = response['answers'][qid]
        if not isinstance(answer, dict) or not isinstance(answer.get('probabilities'), dict):
            raise ServiceError('invalid_output', f'invalid answer for {qid}', 502)
        probs = answer.get('probabilities', {})
        if (answer.get('type') != kind or set(probs) != set(options)
                or set(answer.get('option_ids', [])) != set(options)
                or len(answer.get('option_ids', [])) != len(options)
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values())
                or not math.isclose(sum(probs.values()), 1, abs_tol=1e-9)):
            raise ServiceError('invalid_output', f'invalid probability distribution for {qid}', 502)
        if kind == 'noul' and answer.get('noul') != probs['true']:
            raise ServiceError('invalid_output', f'inconsistent Boolean probability for {qid}', 502)


class DecisionService:
    def __init__(self, config, journal):
        self.config = config
        self.models = config['models']
        self.worker = Path(config['worker']).resolve()
        self.worker_sha = file_hash(self.worker)
        if self.worker_sha != config['worker_sha256']:
            raise ValueError('worker hash mismatch')
        self.journal = Path(journal).resolve()
        self.journal.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.mutex = threading.Lock()
        self.proc = None
        self.active = None
        self.phase = 'unloaded'
        self.ready = None
        self.gpu_lock = None
        self.log = None
        self._verified = {}
        for name, spec in self.models.items():
            if spec['bundle_hash'] != digest(spec['manifest'], canonical=True):
                raise ValueError(f'bundle hash mismatch: {name}')
            if not {'config.json', 'tokenizer.json', 'decision_policy.json', 'model.safetensors'} <= set(spec['manifest']['files_sha256']):
                raise ValueError(f'complete converted checkpoint hashes required: {name}')
            if spec.get('precision', 'bfloat16') not in ('bfloat16', 'float32') or spec.get('backend', 'metal') not in ('metal', 'cpu'):
                raise ValueError(f'unsupported precision/backend: {name}')

    def catalog(self):
        return {'models': [{k: v for k, v in spec.items() if k != 'directory'} | {'id': name}
                           for name, spec in self.models.items()],
                'runtime_code_license': 'MIT', 'worker_sha256': self.worker_sha,
                'execution': {'engine': 'uzu', 'resident_mode': 'one_model',
                              'question_mode': 'sequential', 'gpu_batching': False,
                              'shared_prefix_cache': False, 'generation': False}}

    def health(self):
        proc = self.proc
        return {'status': self.phase, 'active_model': self.active,
                'pid': proc.pid if proc else None,
                'worker_alive': bool(proc and proc.poll() is None),
                'memory': memory(proc.pid) if proc and proc.poll() is None else None,
                'worker_sha256': self.worker_sha, 'ready': self.ready}

    def _save(self, path, body):
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            stream.write(encoded(body)); stream.flush(); os.fsync(stream.fileno())
        tmp.replace(path)

    def _verify(self, spec):
        root = Path(spec['directory']).resolve()
        for name, expected in spec['manifest']['files_sha256'].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ServiceError('invalid_configuration', 'artifact escapes model directory', 500)
            stat = path.stat()
            fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if self._verified.get(str(path)) != (fingerprint, expected):
                if file_hash(path) != expected:
                    raise ServiceError('artifact_changed', f'checkpoint hash mismatch: {name}', 409)
                self._verified[str(path)] = (fingerprint, expected)

    def _unload(self):
        try:
            if self.proc:
                try:
                    self.proc.stdin.close()
                except BrokenPipeError:
                    pass
                self.proc.wait()  # drain naturally; no generation timeout
        finally:
            if self.log:
                self.log.close()
            if self.gpu_lock:
                fcntl.flock(self.gpu_lock, fcntl.LOCK_UN); self.gpu_lock.close()
            self.proc = self.log = self.gpu_lock = self.active = self.ready = None
            self.phase = 'unloaded'

    def unload(self):
        with self.mutex:
            self._unload()
        return self.health()

    def _load(self, name):
        if name not in self.models:
            raise ServiceError('unknown_model', name, 404)
        if self.active == name and self.proc and self.proc.poll() is None:
            return False
        self._unload()
        spec = self.models[name]
        self.phase = 'verifying_artifacts'
        try:
            self._verify(spec)
            if file_hash(self.worker) != self.worker_sha:
                raise ServiceError('artifact_changed', 'worker binary changed', 409)
            if spec.get('backend', 'metal') == 'metal':
                self.phase = 'waiting_for_gpu'
                self.gpu_lock = open(self.config['gpu_lock'], 'a+')
                fcntl.flock(self.gpu_lock, fcntl.LOCK_EX)
            self.phase = 'loading'
            self.log = (self.journal / f'worker-{time.time_ns()}.log').open('w')
            args = [str(self.worker), spec['directory']]
            if spec.get('precision') == 'float32':
                args.append('--f32')
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, env={**os.environ, 'UZU_BACKEND': spec.get('backend', 'metal')})
            line = self.proc.stdout.readline()
            if not line:
                raise ServiceError('load_failed', 'uzu worker exited before ready; see worker log', 503)
            self.ready = strict_json(line)
            if (not isinstance(self.ready, dict) or self.ready.get('event') != 'ready'
                    or self.ready.get('protocol') != 'uzu.typed_decision/1'
                    or not isinstance(self.ready.get('metadata'), dict)
                    or self.ready['metadata'].get('architecture_id') != spec['manifest']['architecture_id']):
                raise ServiceError('invalid_worker', 'unexpected ready protocol/architecture', 502)
            self.active, self.phase = name, 'loaded'
            return True
        except Exception:
            self._unload()
            raise

    def load(self, name):
        with self.mutex:
            cold = self._load(name)
        return self.health() | {'loaded_now': cold}

    def decide(self, body):
        if (not isinstance(body, dict) or set(body) != {'request_id', 'model', 'binding', 'state', 'questions'}
                or not isinstance(body['request_id'], str) or not body['request_id']
                or not isinstance(body['model'], str) or not isinstance(body['binding'], dict)
                or not isinstance(body['questions'], dict) or not body['questions']):
            raise ServiceError('invalid_request', 'expected request_id, model, binding, state and nonempty questions')
        name = body['model']
        if name not in self.models:
            raise ServiceError('unknown_model', str(name), 404)
        if body['binding'].get('bundle_hash') != self.models[name]['bundle_hash']:
            raise ServiceError('binding_mismatch', 'request must bind the configured model bundle', 409)
        # Preserve input/option ordering in this digest: order affects these models.
        fingerprint = digest(body)
        path = self.journal / (hashlib.sha256(body['request_id'].encode()).hexdigest() + '.json')
        queued = time.perf_counter()
        with self.mutex:
            if path.exists():
                old = strict_json(path.read_bytes())
                if old['request_sha256'] != fingerprint:
                    raise ServiceError('request_collision', 'request ID is already bound to another ordered payload', 409)
                if old['phase'] != 'complete':
                    raise ServiceError('outcome_unknown', 'prior request has no terminal receipt; explicit reconciliation required', 409)
                return old['result'] | {'replayed': True}
            self._save(path, {'phase': 'pending', 'request_sha256': fingerprint, 'request_id': body['request_id']})
            started = time.perf_counter()
            before = power()
            telemetry = {'queue_seconds': started - queued}
            result = {'request_id': body['request_id'], 'request_sha256': fingerprint,
                      'binding': body['binding'], 'bundle_hash': self.models[name]['bundle_hash'],
                      'model_id': name, 'worker_sha256': self.worker_sha, 'replayed': False}
            result['runtime'] = {k: self.models[name].get(k, default)
                                 for k, default in [('backend', 'metal'), ('precision', 'bfloat16')]}
            result['runtime']['configuration_sha256'] = digest(self.models[name], canonical=True)
            try:
                loading = time.perf_counter()
                cold = self._load(name)
                telemetry |= {'cold': cold, 'load_seconds': time.perf_counter() - loading if cold else 0}
                request = {k: body[k] for k in ('state', 'questions')}
                self.phase = 'inferencing'
                inference = time.perf_counter()
                with MemorySample(self.proc.pid) as sample:
                    self.proc.stdin.write(encoded(request).decode() + '\n'); self.proc.stdin.flush()
                    line = self.proc.stdout.readline()
                telemetry |= {'inference_seconds': time.perf_counter() - inference, 'memory': sample.result()}
                if not line:
                    raise ServiceError('worker_exited', 'uzu worker exited without an answer', 503)
                raw = strict_json(line)
                if not isinstance(raw, dict):
                    raise ServiceError('invalid_output', 'worker envelope must be an object', 502)
                if raw.get('ok') is True:
                    validate_response(request, raw['response'])
                    result |= {'status': 'ok', 'response': raw['response']}
                elif raw.get('ok') is False and isinstance(raw.get('error'), dict):
                    result |= {'status': raw['error']['code'], 'error': raw['error']}
                else:
                    raise ServiceError('invalid_output', 'invalid worker envelope', 502)
                if result['status'] == 'inference_failed':
                    self._unload()
                else:
                    self.phase = 'loaded'
            except (OSError, ValueError, KeyError, TypeError, ServiceError) as error:
                typed = error if isinstance(error, ServiceError) else ServiceError('inference_failed', str(error), 503)
                result |= {'status': typed.code, 'error': typed.body()}
                self._unload()
            telemetry |= {'wall_seconds': time.perf_counter() - started, 'power_before': before, 'power_after': power(),
                          'execution': 'uzu; sequential questions; no generated text'}
            result['telemetry'] = telemetry
            result['receipt_sha256'] = digest(result)
            self._save(path, {'phase': 'complete', 'request_sha256': fingerprint, 'result': result})
            return result


def http_server(service, port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # no state or request data in access logs

        def send_json(self, status, body):
            content = encoded(body)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers(); self.wfile.write(content)

        def do_GET(self):
            if self.path == '/healthz':
                self.send_json(200, service.health())
            elif self.path == '/v1/models':
                self.send_json(200, service.catalog())
            else:
                self.send_json(404, {'error': {'code': 'not_found'}})

        def do_POST(self):
            try:
                if self.headers.get('Origin') or self.headers.get('Content-Type') != 'application/json':
                    raise ServiceError('invalid_request', 'local JSON clients only; browser origins are not accepted', 403)
                if self.headers.get('Transfer-Encoding'):
                    raise ServiceError('invalid_request', 'send JSON with Content-Length', 411)
                length = int(self.headers['Content-Length'])
                if length < 0:
                    raise ValueError('negative content length')
                body = strict_json(self.rfile.read(length))
                if self.path == '/v1/decisions':
                    result = service.decide(body)
                elif self.path == '/v1/models/load' and set(body) == {'model'}:
                    result = service.load(body['model'])
                elif self.path == '/v1/models/unload' and body == {}:
                    result = service.unload()
                else:
                    raise ServiceError('invalid_request', 'unknown route or lifecycle request')
                self.send_json(200, result)
            except ServiceError as error:
                self.send_json(error.status, {'error': error.body()})
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, {'error': {'code': 'invalid_request', 'message': str(error)}})
            except OSError as error:
                self.send_json(503, {'error': {'code': 'service_unavailable', 'message': str(error)}})
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)
