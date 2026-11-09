#!/usr/bin/env python3
"""Stream request-count scaling with a bounded time-ordering heap.

No giant prefix trie or whole-trace Python list is constructed. Hash namespaces
are independent per copy; theoretical reuse is not recomputed here. Actual cache
reuse is reported by the replay. At alpha <= 1, selected arrivals are unchanged.
"""
import argparse
import hashlib
import heapq
import json
import math
import resource
import time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def uniform(key):
    return (int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')+.5)/2**64


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--seed',type=int,default=20261007)
    p.add_argument('--jitter-seconds',type=float,default=30.)
    p.add_argument('--window-start-seconds',type=float,default=0.)
    p.add_argument('--window-duration-seconds',type=float)
    p.add_argument('--max-output-requests',type=int,default=10000000)
    p.add_argument('--max-wall-seconds',type=float,default=180.)
    p.add_argument('--max-rss-gib',type=float,default=8.)
    p.add_argument('--output-format',choices=['mooncake','dynamo'],default='mooncake')
    a=p.parse_args(); start=time.monotonic()
    base_manifest=a.manifest or a.base.with_suffix(a.base.suffix+'.manifest.json')
    m=json.loads(base_manifest.read_text())
    dataset=m.get('dataset',m.get('dataset_id'))
    full_duration=float(m.get('duration_seconds',m.get('test_duration_seconds')))
    begin=a.window_start_seconds
    duration=a.window_duration_seconds or (full_duration-begin)
    end=begin+duration
    assert 0<a.alpha<=100000 and 0<=begin<end<=full_duration+1e-5
    assert a.jitter_seconds>=0
    stride=int(m.get('namespace_stride',m.get('max_hash_id',0)+1))
    assert stride>0 and math.ceil(a.alpha)*stride<=2**32, 'copy namespace exceeds u32; thin or remap base namespaces'
    block=int(m.get('trace_block_size',m.get('block_size',16)))
    whole=math.floor(a.alpha);fraction=a.alpha-whole
    rag=dataset=='ragpulse'
    # RAG uses its established session phase, including its first copy.
    jitter=a.jitter_seconds if rag or a.alpha>1 else 0.
    if rag and a.seed==20261007:a.seed=20260928
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix(a.output.suffix+'.tmp')
    sha=hashlib.sha256(); seen_count=source_count=count=0
    input_tokens=output_tokens=0;first=last=None;max_rounding_delta=0.
    heap=[];source_previous=-float('inf');peak_heap=0
    source_origin=float(m.get('source_origin_seconds',m.get('test_start_seconds',0)))
    def guard():
        if time.monotonic()-start>a.max_wall_seconds:raise TimeoutError('trace preparation wall guard')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>a.max_rss_gib*1024**2:raise MemoryError('trace preparation RSS guard')
    def emit(f,row):
        nonlocal count,input_tokens,output_tokens,first,last,max_rounding_delta
        if count>=a.max_output_requests:raise RuntimeError('max_output_requests guard')
        if count%4096==0:guard()
        timestamp=row['timestamp']
        if a.output_format=='dynamo':
            timestamp=int(timestamp+.5)
            max_rounding_delta=max(max_rounding_delta,abs(timestamp-row['timestamp']))
            serialized={'schema':'dynamo.request.trace.v1','event_type':'request_end',
                        'event_time_unix_ms':timestamp,
                        'request':{'request_id':f'request-{count:012d}',
                                   'request_received_ms':timestamp,'output_tokens':row['output_length'],
                                   'replay':{'trace_block_size':block,'input_length':row['input_length'],
                                             'input_sequence_hashes':row['hash_ids']}}}
        else:serialized=row
        line=(json.dumps(serialized,separators=(',',':'))+'\n').encode()
        f.write(line);sha.update(line);count+=1
        input_tokens+=row['input_length'];output_tokens+=row['output_length']
        if first is None:first=timestamp
        last=timestamp
    try:
        with a.base.open() as src,temp.open('wb') as out:
            for index,line in enumerate(src):
                if not line.strip():continue
                row=json.loads(line);t=float(row['timestamp'])/1000
                assert t>=source_previous,'base trace must be sorted'
                source_previous=t;seen_count+=1
                if index%4096==0:guard()
                if begin<=t<end or (t==end and end==full_duration):source_count+=1
                if t<begin-jitter:continue
                if t>end+jitter:break
                source_id=row.get('request_id',str(index))
                # As source times advance, all future shifted rows lie >= t-jitter.
                while heap and heap[0][0]<t-jitter:
                    _,_,_,ready=heapq.heappop(heap);emit(out,ready)
                for copy in range(math.ceil(a.alpha)):
                    if copy%4096==0:guard()
                    if copy==whole and uniform(f'{a.seed}:thin:{source_id}')>=fraction:continue
                    shifted=t
                    if jitter:
                        if rag:
                            session=row.get('source_session_id',row.get('metadata',{}).get('source_session_id',source_id))
                            phase=jitter*(2*uniform(f'{a.seed}:copy:{copy}:session:{session}')-1)
                            t_day=(source_origin+t)%86400
                            taper=min(1.,t_day/max(60.,2*jitter),(86400-t_day)/max(60.,2*jitter))
                        else:
                            # Keep the original copy unchanged; independent added copies shift.
                            phase=0. if copy==0 else jitter*(2*uniform(f'{a.seed}:copy:{copy}')-1)
                            taper=min(1.,t/max(60.,2*jitter),(full_duration-t)/max(60.,2*jitter))
                        shifted=t+phase*taper
                    if shifted<begin or shifted>end or (shifted==end and end!=full_duration):continue
                    hashes=row.get('hash_ids',[])
                    ready={'request_id':f'{dataset}:c{copy}:{source_id}',
                           'timestamp':round((shifted-begin)*1000,6),
                           'input_length':int(row['input_length']),
                           'output_length':int(row['output_length']),
                           'hash_ids':[int(h)+copy*stride for h in hashes]}
                    heapq.heappush(heap,(shifted,copy,index,ready))
                peak_heap=max(peak_heap,len(heap))
            while heap:
                _,_,_,ready=heapq.heappop(heap);emit(out,ready)
        assert count,'no requests after thinning'
        temp.replace(a.output)
        manifest={**m,'dataset':dataset,'alpha':a.alpha,'seed':a.seed,
                  'requests':count,'duration_seconds':duration,
                  'source_origin_seconds':source_origin+begin,'trace_block_size':block,
                  'namespace_stride':stride,'base_sha256':m.get('output_sha256',m.get('sha256')),
                  'base_manifest_sha256':digest(base_manifest),
                  'output_sha256':sha.hexdigest(),'mean_rps':count/duration,
                  'mean_isl':input_tokens/count,'mean_osl':output_tokens/count,
                  'first_arrival_ms':first,'last_arrival_ms':last,
                  'source_requests_in_window':source_count,'base_records_scanned':seen_count,
                  'window_start_seconds_in_base':begin,'window_end_seconds_in_base':end,
                  'window_kind':'formal_full' if begin==0 and end==full_duration else 'calibration_slice',
                  'trace_format':a.output_format,'format':a.output_format,
                  'timestamp_quantization_ms':1 if a.output_format=='dynamo' else None,
                  'maximum_timestamp_rounding_delta_ms':max_rounding_delta,
                  'carrier_event_note':'Synthetic request_end at arrival is a replay carrier, not measured production completion' if a.output_format=='dynamo' else None,
                  'jitter_seconds':jitter,'copies_independent_hashes':True,
                  'phase_policy':'ragpulse_legacy_session_day_taper' if rag else 'original_copy_zero_phase_added_copies_fixed_phase_edge_taper',
                  'theoretical_prefix_reuse':'not_recomputed_bounded_memory; see actual replay reuse',
                  'preparation_wall_s':time.monotonic()-start,
                  'preparation_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'preparation_peak_heap_rows':peak_heap,
                  'scaler_sha256':digest(__file__)}
        a.output.with_suffix(a.output.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('dataset','alpha','requests','duration_seconds','mean_rps','output_sha256','preparation_wall_s','preparation_peak_rss_kib')},indent=2))
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


if __name__=='__main__':main()
