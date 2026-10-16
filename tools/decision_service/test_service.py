"""Transport/lifecycle failure scenarios. Fake worker is explicit; no quality claim."""
import json
from pathlib import Path
import threading
import urllib.request
import urllib.error
import pytest
from decision_service.server import DecisionService, http_server, file_hash, digest


@pytest.fixture
def service(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    files = {}
    for name in ['config.json', 'tokenizer.json', 'decision_policy.json', 'model.safetensors']:
        path = model / name; path.write_text('{}'); files[name] = file_hash(path)
    worker = tmp_path / 'worker'
    worker.write_text('''#!/usr/bin/env python3
import json,sys,os
print(json.dumps({'event':'ready','protocol':'uzu.typed_decision/1','metadata':{'architecture_id':'test_fixture'}}),flush=True)
calls=0
for line in sys.stdin:
 r=json.loads(line);calls+=1
 if r['state']=='crash':os._exit(2)
 if r['state']=='nonobject':print('[]',flush=True);continue
 if r['state']=='overflow':
  print(json.dumps({'ok':False,'error':{'code':'context_exceeded','actual_tokens':999,'maximum_tokens':512,'truncated':False}}),flush=True);continue
 a={}
 for name,q in r['questions'].items():
  k=list(q['criteria']) if q['type']=='choice' else ['false','true']
  p={v:1/len(k) for v in k}
  if r['state']=='bad':p[k[0]]=float('nan')
  a[name]={'type':q['type'],'option_ids':k,'probabilities':p,'input_tokens':calls}
  if q['type']=='noul':a[name]['noul']=p['true']
 print(json.dumps({'ok':True,'response':{'answers':a}}),flush=True)
''')
    worker.chmod(0o700)
    manifest={'architecture_id':'test_fixture','files_sha256':files,'weights_license':'synthetic fixture','revision':'fixture'}
    config={'worker':str(worker),'worker_sha256':file_hash(worker),'gpu_lock':str(tmp_path/'gpu.lock'),
            'models':{'test':{'directory':str(model),'manifest':manifest,'bundle_hash':digest(manifest,canonical=True)}}}
    s=DecisionService(config,tmp_path/'receipts');server=http_server(s);thread=threading.Thread(target=server.serve_forever);thread.start()
    origin=f'http://127.0.0.1:{server.server_port}'
    def call(route,body=None):
        request=urllib.request.Request(origin+route,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request) as response:return json.load(response)
    yield s,call,origin
    server.shutdown();server.server_close();thread.join();s.unload()


def body(service, state='ordinary', request_id='one'):
    return {'request_id':request_id,'model':'test','binding':{'bundle_hash':service.models['test']['bundle_hash']},
            'state':state,'questions':{'q':{'type':'choice','instructions':'Choose','criteria':{'a':'A','b':'B'}}}}


def test_http_replay_binding_restart_and_unload(service):
    s,call,_=service;request=body(s)
    first=call('/v1/decisions',request);assert first['status']=='ok'
    repeat=call('/v1/decisions',request)
    assert repeat['replayed'] and repeat['receipt_sha256']==first['receipt_sha256']
    assert repeat['response']['answers']['q']['input_tokens']==1 # fixture counts actual worker calls
    changed=body(s,'different')
    with pytest.raises(urllib.error.HTTPError) as e:call('/v1/decisions',changed)
    assert e.value.code==409
    reordered=body(s);reordered['questions']['q']['criteria']={'b':'B','a':'A'}
    with pytest.raises(urllib.error.HTTPError):call('/v1/decisions',reordered)
    assert call('/v1/models/unload',{})['pid'] is None
    # A new service object over the technical journal returns the original receipt.
    restored=DecisionService(s.config,s.journal)
    assert restored.decide(request)['replayed'] and restored.proc is None


def test_overflow_and_crash_are_not_model_scores(service):
    s,call,_=service
    overflow=call('/v1/decisions',body(s,'overflow','over'))
    assert overflow['status']=='context_exceeded' and 'response' not in overflow
    assert overflow['error']['truncated'] is False
    failed=call('/v1/decisions',body(s,'crash','dead'))
    assert failed['status']=='worker_exited' and 'response' not in failed
    assert call('/healthz')['pid'] is None
    assert call('/v1/decisions',body(s,request_id='new'))['status']=='ok'


def test_invalid_worker_probabilities_are_rejected(service):
    s,call,_=service
    result=call('/v1/decisions',body(s,'bad'))
    assert result['status']!='ok' and 'response' not in result


def test_nonobject_worker_envelope_is_recorded_and_next_request_recovers(service):
    s,call,_=service
    result=call('/v1/decisions',body(s,'nonobject'))
    assert result['status']=='invalid_output' and 'response' not in result
    assert call('/healthz')['pid'] is None
    assert call('/v1/decisions',body(s,request_id='recovered'))['status']=='ok'


def test_multi_question_and_artifact_integrity(service):
    s,call,_=service;r=body(s)
    r['questions']['stage_stated']={'type':'noul','instructions':'Stage present?','criteria':None}
    result=call('/v1/decisions',r)
    assert set(result['response']['answers'])=={'q','stage_stated'}
    call('/v1/models/unload',{})
    (Path(s.models['test']['directory'])/'config.json').write_text('changed')
    result=call('/v1/decisions',body(s,request_id='changed-model'))
    assert result['status']=='artifact_changed' and s.proc is None


def test_pending_receipt_never_silently_reexecutes(service):
    s,call,_=service;r=body(s)
    import hashlib
    path=s.journal/(hashlib.sha256(r['request_id'].encode()).hexdigest()+'.json')
    s._save(path,{'phase':'pending','request_sha256':digest(r)})
    with pytest.raises(urllib.error.HTTPError) as e:call('/v1/decisions',r)
    assert e.value.code==409 and s.proc is None


def test_parallel_http_calls_do_not_interleave_worker_lines(service):
    from concurrent.futures import ThreadPoolExecutor
    s,call,_=service
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda i:call('/v1/decisions',body(s,request_id=f'p{i}')),range(4)))
    assert all(r['status']=='ok' for r in rows)
    assert sorted(r['response']['answers']['q']['input_tokens'] for r in rows)==[1,2,3,4]


def test_malformed_http_request_does_not_load_model(service):
    s,call,origin=service;r=body(s);r['binding']=None
    with pytest.raises(urllib.error.HTTPError) as e:call('/v1/decisions',r)
    assert e.value.code==400 and s.proc is None
    req=urllib.request.Request(origin+'/v1/decisions',data=b'{"model":"a","model":"b"}',headers={'Content-Type':'application/json'})
    with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(req)
    assert s.proc is None
