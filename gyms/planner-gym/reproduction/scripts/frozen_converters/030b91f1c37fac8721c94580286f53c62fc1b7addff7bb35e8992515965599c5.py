#!/usr/bin/env python3
"""Canonical 1x formal-test traces; run only inside allocated CPU nodes."""
import argparse, collections, datetime, hashlib, json, math, time
from pathlib import Path

DAY=86400

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def verified_sources(root,meta):
 answer=[]
 for s in meta.get('source_files',[]):
  p=root/'source'/s['file'];actual=sha(p)
  assert actual==s['sha256'],(str(p),actual,s['sha256'])
  answer.append(dict(s,local_path=str(p),verified_sha256=actual))
 return answer

class Writer:
 def __init__(self,out,ident,block):
  self.path=out/ident/'test-1x.jsonl';self.path.parent.mkdir(parents=True,exist_ok=True)
  self.temp=self.path.with_suffix('.jsonl.tmp');self.f=self.temp.open('w');self.ident=ident;self.block=block
  self.n=self.si=self.so=self.maxhash=self.max_isl=self.max_osl=0;self.first=None;self.last=-1.;self.hash_mapping={};self.opaque_next=1;self.hourly=collections.Counter();self.five_min=collections.Counter()
 def emit(self,t,inp,out,ids,source_row,extra=None):
  inp,out=int(inp),int(out);t=round(float(t),6)
  assert inp>=0 and out>=0 and t>=self.last,(self.ident,t,self.last,inp,out)
  assert len(ids)==math.ceil(inp/self.block),(self.ident,source_row,len(ids),inp,self.block)
  assert all(isinstance(h,int) and 0<=h<2**32 for h in ids)
  row={'request_id':f'{self.ident}:r{source_row}','timestamp':t,'input_length':inp,'output_length':out,'hash_ids':ids,'source_row':int(source_row)}
  if extra:row.update(extra)
  assert 'session_id' not in row
  self.f.write(json.dumps(row,separators=(',',':'))+'\n')
  self.n+=1;self.si+=inp;self.so+=out;self.max_isl=max(self.max_isl,inp);self.max_osl=max(self.max_osl,out)
  if ids:self.maxhash=max(self.maxhash,max(ids))
  if self.first is None:self.first=t
  self.last=t;self.hourly[int(t//3600000)]+=1;self.five_min[int(t//300000)]+=1
 def opaque(self,inp):
  n=math.ceil(int(inp)/self.block);start=self.opaque_next;self.opaque_next+=n
  assert self.opaque_next<2**32
  return list(range(start,start+n))
 def mapped(self,ids):
  result=[]
  for h in ids:
   key=(type(h).__name__,str(h))
   if key not in self.hash_mapping:self.hash_mapping[key]=len(self.hash_mapping)+1
   result.append(self.hash_mapping[key])
  return result
 def finish(self,meta):
  self.f.close();assert self.n>0;self.temp.replace(self.path)
  m=dict(meta,dataset=self.ident,alpha=1,requests=self.n,mean_isl=self.si/self.n,mean_osl=self.so/self.n,max_isl=self.max_isl,max_osl=self.max_osl,total_input_tokens=self.si,total_output_tokens=self.so,first_arrival_ms=self.first,last_arrival_ms=self.last,trace_block_size=self.block,namespace_stride=self.maxhash+1,max_hash_id=self.maxhash,output_sha256=sha(self.path),output_bytes=self.path.stat().st_size,output_path=str(self.path),format='mooncake',timestamps='milliseconds relative to formal test window origin; idle gaps preserved; no time compression',sorted_by_timestamp=True,hash_dtype='u32',source_row_identity_stable=True,has_session_id=False,converter_sha256=sha(__file__))
  m['mean_rps']=self.n/m['duration_seconds'];m['source_requests']=self.n
  m['hourly_request_counts']=[self.hourly.get(i,0) for i in range(max(1,math.ceil(m['duration_seconds']/3600)))];m['five_minute_request_counts']=[self.five_min.get(i,0) for i in range(max(1,math.ceil(m['duration_seconds']/300)))];m['busy_hour_start_seconds']=max(self.hourly,key=lambda k:self.hourly[k])*3600;m['busy_hour_requests']=max(self.hourly.values())
  atomic_json(self.path.with_suffix('.jsonl.manifest.json'),m)
  return m

def prepare_short(ident,root,out,meta):
 srcs=verified_sources(root,meta);p=Path(srcs[0]['local_path'])
 rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
 origin=min(float(r['timestamp']) for r in rows)
 indexed=sorted(enumerate(rows),key=lambda x:(float(x[1]['timestamp']),x[0]))
 w=Writer(out,ident,meta['block_size'])
 for i,r in indexed:
  ids=r['hash_ids'];native=all(isinstance(h,int) and 0<=h<2**32 for h in ids)
  if not native:ids=w.mapped(ids)
  extra={}
  for key in ['chat_id','parent_chat_id','turn']:
   if key in r:extra['source_'+key]=r[key]
  w.emit(float(r['timestamp'])-origin,r['input_length'],r['output_length'],ids,i,extra)
 duration=(max(float(r['timestamp']) for r in rows)-origin)/1000
 return w.finish(dict(meta,source_files=srcs,source_origin_seconds=origin/1000,window_start_seconds=0,window_end_seconds=duration,test_start_seconds=0,test_end_seconds=duration,duration_seconds=duration,test_includes_final_timestamp=True,transform='identity length/native hash values; stable arrival sort and origin normalization; no warmup',hash_assumption='native block-hash identity and prefix structure preserved'))

def prepare_rag(root,out,meta):
 import ragpulse_converter as c
 c.SOURCES={p.name:p for p in sorted((root/'source/ragpulse').glob('*.jsonl'))}
 base,stride=c.build_base(64)
 w=Writer(out,'ragpulse',64)
 for r in base:
  if 4*DAY<=r['source_timestamp_s']<5*DAY:
   w.emit((r['source_timestamp_s']-4*DAY)*1000,r['input_length'],r['output_length'],r['hash_ids'],r['source_row'],{'source_timestamp_s':r['source_timestamp_s'],'source_session_id':r['source_session_id'],'unknown_tail_tokens':r['unknown_tail_tokens']})
 m=w.finish(dict(meta,source_files=[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'url':f"https://raw.githubusercontent.com/flashserve/RAGPulse/{meta['revision']}/data/{p.name}"} for p in c.SOURCES.values()],source_origin_seconds=4*DAY,window_start_seconds=4*DAY,window_end_seconds=5*DAY,test_start_seconds=4*DAY,test_end_seconds=5*DAY,duration_seconds=DAY,test_includes_final_timestamp=False,transform='opaque component token spans mapped to cumulative 64-token blocks; original five-day dictionary; select day5 unshifted 1x; 0-12 unknown template tokens use request-unique identity',hash_assumption='component approximation, not exact original token identities',scaling_phase_seed=20260928,scaling_phase_mode='session',scaling_jitter_seconds=30,base_is_unjittered=True,legacy_converter_sha256=sha(c.__file__)))
 # Keep full first-five-day ID namespace so arbitrary multiplicities preserve previous scaling.
 m['namespace_stride']=stride;atomic_json(w.path.with_suffix('.jsonl.manifest.json'),m);return m

def prepare_csv(ident,root,out,meta):
 import pandas as pd
 import numpy as np
 srcs=verified_sources(root,meta);p=Path(srcs[0]['local_path'])
 azure=ident.startswith('azure');begin,end=(4*DAY,5*DAY) if azure else (7*DAY,14*DAY)
 cols=['TIMESTAMP','ContextTokens','GeneratedTokens'] if azure else ['Timestamp','Request tokens','Response tokens']
 arrays=[];source_index=0;origin=None;source_min=float('inf');scanned=0
 for chunk in pd.read_csv(p,usecols=cols,chunksize=250000):
  if azure:
   ts=pd.to_datetime(chunk['TIMESTAMP'],utc=True,format='mixed').astype('int64').to_numpy()
   if origin is None:origin=int(ts.min())
   rel=(ts-origin)/1e9
   source_min=min(source_min,int(ts.min()))
  else:
   ts=chunk['Timestamp'].to_numpy(dtype=float)
   if origin is None:origin=float(meta['full_trace_first_timestamp'])
   rel=ts-origin;source_min=min(source_min,float(ts.min()))
  mask=(rel>=begin)&(rel<end)
  inds=np.arange(source_index,source_index+len(chunk),dtype=np.int64)[mask]
  inp=chunk[cols[1]].to_numpy(dtype=np.int64)[mask];osl=chunk[cols[2]].to_numpy(dtype=np.int64)[mask]
  arrays.append((rel[mask],inp,osl,inds));source_index+=len(chunk);scanned+=len(chunk)
 assert source_min==origin,(ident,source_min,origin)
 ts,inp,osl,idx=[np.concatenate([a[k] for a in arrays]) for k in range(4)]
 order=np.argsort(ts,kind='stable');w=Writer(out,ident,meta['block_size'])
 for j in order:
  w.emit((ts[j]-begin)*1000,inp[j],osl[j],w.opaque(inp[j]),idx[j])
 return w.finish(dict(meta,source_files=srcs,source_full_rows_scanned=scanned,source_first_timestamp_ns=int(origin) if azure else None,source_first_timestamp_seconds=float(origin) if not azure else None,source_origin_seconds=begin,window_start_seconds=begin,window_end_seconds=end,test_start_seconds=begin,test_end_seconds=end,duration_seconds=end-begin,test_includes_final_timestamp=False,transform='select formal test window; stable timestamp sort; preserve lengths; independently unique block IDs for every input block of every request',has_source_hashes=False,hash_assumption='zero prefix reuse assumption: source does not publish hashes; unique opaque IDs do not claim observed absence of reuse'))

def prepare_art(root,out,meta):
 import fsspec, pyarrow as pa, pyarrow.parquet as pq
 import concurrent.futures
 url=f"https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M/resolve/{meta['revision']}/parquet/20260405.parquet"
 projected=root/'source/artchat-20260405-with-hashes.parquet'
 remote={}
 if not projected.exists():
  fs=fsspec.filesystem('http')
  with fs.open(url,'rb',block_size=65536,cache_type='readahead') as f:
   pp=pq.ParquetFile(f);md=pp.metadata;size=f.size
  remote={'full_remote_file_bytes':size,'row_groups':md.num_row_groups,'rows':md.num_rows,'columns_downloaded':['request_id','timestamp_ms','input_length','output_length','hash_ids'],'messages_column_downloaded':False}
  def read_group(i):
   with fs.open(url,'rb',block_size=65536,cache_type='readahead',size=size) as f:
    return pq.ParquetFile(f,metadata=md).read_row_group(i,columns=remote['columns_downloaded'],use_threads=False)
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   writer=None
   try:
    for i,table in enumerate(pool.map(read_group,range(md.num_row_groups))):
     if writer is None:writer=pq.ParquetWriter(projected,table.schema)
     writer.write_table(table)
     print(json.dumps({'art_row_group_complete':i,'rows':table.num_rows}),flush=True)
   finally:
    if writer:writer.close()
  atomic_json(projected.with_suffix('.source.json'),remote)
 else:remote=json.loads(projected.with_suffix('.source.json').read_text())
 w=Writer(out,'art-chat',256);source_i=0;begin,end=4*DAY,5*DAY
 for batch in pq.ParquetFile(projected).iter_batches(batch_size=4096):
  for r in batch.to_pylist():
   ts=r['timestamp_ms']/1000
   if begin<=ts<end:w.emit(r['timestamp_ms']-begin*1000,r['input_length'],r['output_length'],w.mapped(r['hash_ids']),source_i,{'source_request_id':r['request_id']})
   source_i+=1
 return w.finish(dict(meta,source_files=[dict(remote,url=url,projection_path=str(projected),projection_sha256=sha(projected),projection_bytes=projected.stat().st_size)],source_origin_seconds=begin,window_start_seconds=begin,window_end_seconds=end,test_start_seconds=begin,test_end_seconds=end,duration_seconds=DAY,test_includes_final_timestamp=False,transform='download real hash_ids and length/timestamp/request ID columns from pinned day5 parquet; exclude messages; bijective string hash to u32 mapping in first-observed order',has_source_hashes=True,hash_assumption='native 256-token prefix identities preserved under bijection',unique_hashes=len(w.hash_mapping)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--root',type=Path,default=Path(__file__).parent);ap.add_argument('--output',type=Path,default=Path('/workspace/data'));a=ap.parse_args()
 started=time.monotonic();catalog=json.loads((a.root/'catalog.json').read_text());meta=catalog[a.dataset]
 if a.dataset=='ragpulse':m=prepare_rag(a.root,a.output,meta)
 elif a.dataset=='art-chat':m=prepare_art(a.root,a.output,meta)
 elif a.dataset.startswith('azure') or a.dataset=='burstgpt-3':m=prepare_csv(a.dataset,a.root,a.output,meta)
 else:m=prepare_short(a.dataset,a.root,a.output,meta)
 m.update(preparation_wall_seconds=time.monotonic()-started,prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_status='ready')
 p=a.output/a.dataset/'test-1x.jsonl.manifest.json';atomic_json(p,m)
 rp=a.output/'source-registry.json';reg=json.loads(rp.read_text()) if rp.exists() else {'schema':'nvopt.canonical-test-traces.v1','datasets':{}}
 reg['datasets'][a.dataset]=dict(m,shared_path=f'/home/scratch.hongkuanz_gpu/nvopt-static-overnight-20261007/data/{a.dataset}/test-1x.jsonl',shared_manifest=f'/home/scratch.hongkuanz_gpu/nvopt-static-overnight-20261007/data/{a.dataset}/test-1x.jsonl.manifest.json')
 atomic_json(rp,reg)
 print(json.dumps({'ready':a.dataset,'requests':m['requests'],'duration_seconds':m['duration_seconds'],'mean_isl':m['mean_isl'],'mean_osl':m['mean_osl'],'sha256':m['output_sha256'],'wall_seconds':m['preparation_wall_seconds']}),flush=True)
if __name__=='__main__':main()
