// Validate current offline pages, including pending replacements, through Chrome DevTools.
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';

const root=path.dirname(fileURLToPath(import.meta.url));
const endpoint=process.env.CHROME_ENDPOINT || 'http://127.0.0.1:9247';
const target=await (await fetch(endpoint+'/json/new?about:blank',{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve=>socket.addEventListener('open',resolve,{once:true}));
let nextId=0;
const pending=new Map(),exceptions=[],checks=[];
socket.addEventListener('message',event=>{
 const value=JSON.parse(event.data);
 if(value.method==='Runtime.exceptionThrown')exceptions.push(value.params.exceptionDetails);
 if(value.id&&pending.has(value.id)){
  const {resolve,reject,timer}=pending.get(value.id);pending.delete(value.id);clearTimeout(timer);
  value.error?reject(new Error(JSON.stringify(value.error))):resolve(value.result);
 }
});
function call(method,params={}){
 return new Promise((resolve,reject)=>{
  const id=++nextId,timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method));},15000);
  pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));
 });
}
async function evaluate(expression){
 const value=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
 assert(!value.exceptionDetails,JSON.stringify(value.exceptionDetails));return value.result.value;
}
async function navigate(filename){
 await call('Page.navigate',{url:pathToFileURL(path.join(root,filename)).href});
 for(let i=0;i<100;i++){
  if(await evaluate(`document.readyState==='complete'&&location.pathname.endsWith('/${filename}')&&typeof report!=='undefined'&&document.getElementById('app').textContent.length>100`))return;
  await new Promise(resolve=>setTimeout(resolve,50));
 }
 throw new Error('Page did not render: '+filename);
}
await call('Runtime.enable');await call('Page.enable');
try{
 for(const filename of ['index.html','aggregated.html','disaggregated.html','latency.html','slo-rankings.html','traces.html']){
  await navigate(filename);
  assert(await evaluate(`document.querySelectorAll('.nav a:not([hidden])').length===6`));
  assert(await evaluate(`Object.keys(report.experiment_setup.budget_mismatches).length===0`));
  assert(await evaluate(`(()=>{const p=report.resource_policy;const key=Number.isInteger(p.gpu_budget_extra_workers)?'baseline-plus-'+p.gpu_budget_extra_workers+'-max-workers':p.gpu_budget_multiplier+'x';return document.querySelector('[data-resource-policy="'+key+'"]')!==null;})()`));
  assert(await evaluate(`Object.values(report.experiment_setup.rows).every(s=>{const p=report.resource_policy;const expected=Number.isInteger(p.gpu_budget_extra_workers)?s.initial_gpus+p.gpu_budget_extra_workers*Math.max(...Object.values(s.gpus_per_worker)):p.gpu_budget_multiplier*s.initial_gpus;return s.gpu_budget===expected&&s.feasible_fleet_count>=1;})`));
  assert(await evaluate(`Object.values(report.pages).every(p=>p.cases.every(c=>Object.values(c.rows).every(r=>r.status!=='pending'||(Object.keys(r.metrics).length===0&&r.source_sha256===null))))`));
  if(['aggregated.html','disaggregated.html'].includes(filename)){
   for(const regime of ['all','HT','LC']){
    await evaluate(`document.querySelector('[data-regime="${regime}"]').click()`);
    const data=await evaluate(`(()=>{const v=activePage.views[state.regime];return {actual:[...document.querySelectorAll('#core-ranking tbody tr')].map(r=>({planner:r.dataset.planner,rank:Number.isNaN(parseFloat(r.cells[0].textContent))?null:parseFloat(r.cells[0].textContent),mean:Number.isNaN(parseFloat(r.querySelector('.rank-value').textContent))?null:parseFloat(r.querySelector('.rank-value').textContent)})),expected:v.core.rows.map(r=>({planner:r.planner,rank:r.rank,mean:r.mean_rank==null?null:Number(r.mean_rank.toFixed(3))})),cases:document.querySelectorAll('#case-rows tr').length}})()`);
    assert.deepEqual(data.actual,data.expected);assert.equal(data.cases,regime==='all'?30:15);
   }
   await evaluate(`document.querySelector('[data-regime="all"]').click();showCase(activePage.cases[0].case_id)`);
   assert(await evaluate(`document.getElementById('case-dialog').open&&document.querySelectorAll('.detail-table tbody tr').length===6`));
   assert(await evaluate(`document.querySelector('#case-content a[href="traces.html#'+activePage.cases[0].case_id+'"]')!==null`));
   await evaluate(`document.getElementById('close-dialog').click();document.getElementById('attention').click()`);
   assert(await evaluate(`document.querySelectorAll('#case-rows [data-case]').length===activePage.cases.filter(isIncomplete).length`));
   await evaluate(`document.getElementById('attention').click();document.getElementById('search').value='no-such-case';document.getElementById('search').dispatchEvent(new Event('input'))`);
   assert(await evaluate(`!!document.querySelector('#case-rows .empty')`));
  }
  if(filename==='latency.html'){
   assert(await evaluate(`document.querySelectorAll('.latency-table').length===2`));
   assert(await evaluate(`latencySummaryCounts().total.count===360`));
   await evaluate(`const b=document.querySelector('[data-diagnose-case]');if(b)b.click()`);
   assert(await evaluate(`document.getElementById('case-dialog').open&&!!document.querySelector('.diagnosis-text').textContent`));
   await evaluate(`document.getElementById('close-dialog').click()`);
  }
  if(filename==='traces.html'){
   // Every trace: the table ranks the saved scores, and the arrival series is the whole trace.
   assert(await evaluate(`document.querySelectorAll('#trace-select option').length===60&&allCases.length===60`));
   const mismatches=await evaluate(`allCases.filter(c=>{
    showTrace(c.case_id,true);
    const rows=[...document.querySelectorAll('#trace-leaderboard tbody tr')],scores=rows.map(r=>c.rows[r.dataset.planner].metrics.goodput_per_gpu);
    const ranks=rows.map(r=>Number(r.dataset.rank)),expected=scores.map(s=>scores.filter(o=>o>s).length+(scores.filter(o=>o===s).length+1)/2);
    const arrivals=traces.arrivals[c.case_id].counts.reduce((a,b)=>a+b,0);
    const drawn=traceState.panels.length>=3&&traceState.panels.every(p=>{const g=p.canvas.getContext('2d').getImageData(0,0,p.canvas.width,p.canvas.height).data;for(let i=3;i<g.length;i+=4)if(g[i])return true;return false;});
    return !(rows.length===6&&scores.every((s,i)=>i===0||scores[i-1]>=s)&&ranks.every((r,i)=>r===expected[i])&&arrivals===c.requests&&location.hash==='#'+c.case_id&&drawn);
   }).map(c=>c.case_id)`);
   assert.deepEqual(mismatches,[]);
   // A planner's line is its own saved telemetry, or absent: nothing is drawn for a run without a series.
   assert(await evaluate(`allCases.every(c=>report.core.every(p=>{const d=traceData(c).lines[p],t=traces.timelines[c.case_id+'-'+p];return t?d.measured&&d.series===t:p==='static'?d&&!d.measured&&Object.keys(d.series).sort().join()==='gpus,ready_decode,ready_prefill':d===undefined;}))`));
   // Controls: hiding a planner, the log scale, zooming and resetting, and following the address.
   const controls=await evaluate(`(async()=>{
    showTrace('D-HT-L01',true);
    const before=traceState.panels.map(p=>p.key).join();
    const toggle=document.querySelector('[data-trace-planner="static"]');toggle.click();
    const hidden=traceState.hidden.has('static')&&traceState.panels.every(p=>p.lines.every(l=>l.planner!=='static'));
    document.querySelector('[data-trace-planner="static"]').click();
    const restored=!traceState.hidden.has('static')&&traceState.panels.map(p=>p.key).join()===before;
    document.getElementById('trace-log').click();const log=traceState.log;document.getElementById('trace-log').click();
    const c=traceCase();traceState.view=[c.duration_s/4,c.duration_s/2];traceBehavior(c);
    const zoomed=!document.getElementById('trace-reset').disabled&&traceAxis(c).x0===c.duration_s/4;
    document.getElementById('trace-reset').click();
    const reset=traceState.view===null&&document.getElementById('trace-reset').disabled;
    location.hash='#A-LC-S10';await new Promise(resolve=>setTimeout(resolve,50));
    const followed=traceState.caseId==='A-LC-S10'&&document.getElementById('trace-select').value==='A-LC-S10';
    document.getElementById('trace-next').click();
    return {hidden,restored,log,zoomed,reset,followed,next:traceState.caseId!=='A-LC-S10'};
   })()`);
   assert.deepEqual(controls,{hidden:true,restored:true,log:true,zoomed:true,reset:true,followed:true,next:true});
  }
  if(filename==='slo-rankings.html')assert(await evaluate(`Object.values(report.pages).every(p=>sloAdjustedRanking(p).case_count===p.views.all.core.shared_cases.length)`));
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate(`new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))`);
  assert(await evaluate(`document.documentElement.scrollWidth<=390`),filename+' mobile overflow');
  await call('Emulation.clearDeviceMetricsOverride');
  checks.push({page:filename,status:'passed'});
 }
 assert.equal(exceptions.length,0,JSON.stringify(exceptions));
 const record={checked_at:new Date().toISOString(),checks,exceptions};
 await writeFile(path.join(root,'browser-validation.json'),JSON.stringify(record,null,2)+'\n');
 console.log(JSON.stringify(record));
}catch(error){console.error(JSON.stringify({exceptions}));throw error;}
finally{await call('Page.close').catch(()=>{});socket.close();}
