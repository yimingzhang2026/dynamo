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
   // Controls: hiding a planner, the value scales, the period shown by button and key, and following the address.
   const controls=await evaluate(`(async()=>{
    showTrace('D-HT-L01',true);
    const before=traceState.panels.map(p=>p.key).join();
    const toggle=document.querySelector('[data-trace-planner="static"]');toggle.click();
    const hidden=traceState.hidden.has('static')&&traceState.panels.every(p=>p.lines.every(l=>l.planner!=='static'));
    document.querySelector('[data-trace-planner="static"]').click();
    const restored=!traceState.hidden.has('static')&&traceState.panels.map(p=>p.key).join()===before;
    document.getElementById('trace-log').click();const log=traceState.log;document.getElementById('trace-log').click();
    const c=traceCase(),full=c.duration_s,id=name=>document.getElementById(name),shown=(from,to)=>{const [x0,x1]=traceWindow();return Math.abs(x0-from)<1e-6*full&&Math.abs(x1-to)<1e-6*full;};
    traceSetView(full/4,full/2);
    const zoomed=!id('trace-reset').disabled&&shown(full/4,full/2)&&id('trace-range').textContent===traceClock(full/4)+' – '+traceClock(full/2)+' ('+traceLength(full/4)+')';
    const top=p=>traceScale(p).ticks.at(-1),fitted=traceState.panels.map(top);
    id('trace-fit').click();const whole=traceState.panels.map(top),unfitted=!traceState.fit;id('trace-fit').click();
    const fit=unfitted&&traceState.fit&&fitted.every((v,i)=>v<=whole[i])&&fitted.some((v,i)=>v<whole[i]);
    id('trace-in').click();const nearer=shown(full*5/16,full*7/16);
    id('trace-out').click();const farther=shown(full/4,full/2);
    id('trace-right').click();const later=shown(full*3/8,full*5/8);
    for(let i=0;i<4;i++)id('trace-left').click();
    const earliest=shown(0,full/4)&&id('trace-left').disabled&&!id('trace-right').disabled;
    const key=name=>id('trace-charts').dispatchEvent(new KeyboardEvent('keydown',{key:name,bubbles:true,cancelable:true}));
    key('ArrowRight');key('-');const keyed=shown(0,full/2);
    key('+');key('+');
    for(let i=0;i<40;i++)id('trace-in').click();
    const [x0,x1]=traceWindow(),least=Math.abs(x1-x0-4*traceState.data.width)<1e-6*full&&id('trace-in').disabled&&x0>=0&&x1<=full;
    // The time axis names the moments it marks, whatever the period.
    const ticks=traceTicks(c,x0,x1,9),labelled=ticks.ticks.length>=2&&ticks.ticks.every(t=>t>=x0-1e-6&&t<=x1+1e-6)&&new Set(ticks.ticks.map(ticks.label)).size===ticks.ticks.length;
    key('0');
    const reset=traceState.view===null&&id('trace-reset').disabled&&id('trace-out').disabled&&id('trace-range').textContent.startsWith('Whole trace');
    location.hash='#A-LC-S10';await new Promise(resolve=>setTimeout(resolve,50));
    const followed=traceState.caseId==='A-LC-S10'&&document.getElementById('trace-select').value==='A-LC-S10';
    document.getElementById('trace-next').click();
    return {hidden,restored,log,zoomed,fit,nearer,farther,later,earliest,keyed,least,labelled,reset,followed,next:traceState.caseId!=='A-LC-S10'};
   })()`);
   assert.deepEqual(controls,{hidden:true,restored:true,log:true,zoomed:true,fit:true,nearer:true,farther:true,later:true,earliest:true,keyed:true,least:true,labelled:true,reset:true,followed:true,next:true});
   // The pointer itself: the wheel zooms around it and moves sideways, a drag picks a period or moves along
   // the trace, the strip's window can be dragged, and a double-click returns to the whole trace.
   await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
   await evaluate(`showTrace('D-HT-L01',true);traceState.panels[0].canvas.scrollIntoView({block:'center'})`);
   await new Promise(resolve=>setTimeout(resolve,400));
   const area=await evaluate(`(()=>{const b=traceState.panels[0].canvas.getBoundingClientRect(),n=document.getElementById('trace-nav').getBoundingClientRect();return {left:b.left+traceMargin.left,width:b.width-traceMargin.left-traceMargin.right,y:b.top+b.height/2,strip:n.top+n.height/2,full:traceCase().duration_s};})()`);
   const at=share=>area.left+area.width*share,view=()=>evaluate(`traceWindow()`);
   const mouse=(type,x,y,options={})=>call('Input.dispatchMouseEvent',{type,x,y,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1,...options});
   const drag=async(from,to,y)=>{await mouse('mousePressed',at(from),y);await mouse('mouseMoved',at((from+to)/2),y);await mouse('mouseMoved',at(to),y);await mouse('mouseReleased',at(to),y);};
   const close=(actual,expected)=>assert(actual.every((v,i)=>Math.abs(v-expected[i])<2e-3*area.full),JSON.stringify({actual,expected}));
   await call('Input.dispatchMouseEvent',{type:'mouseWheel',x:at(.25),y:area.y,deltaX:0,deltaY:-120});
   let shown=await view(),span=shown[1]-shown[0];
   assert(span<area.full&&span>area.full/2);close([shown[0]+span/4],[area.full/4]);
   await call('Input.dispatchMouseEvent',{type:'mouseWheel',x:at(.25),y:area.y,deltaX:area.width/10,deltaY:0});
   close(await view(),[shown[0]+span/10,shown[1]+span/10]);
   shown=await view();
   await drag(.25,.75,area.y);
   close(await view(),[shown[0]+span/4,shown[0]+span*3/4]);
   shown=await view();span=shown[1]-shown[0];
   await evaluate(`document.querySelector('[data-trace-mode="pan"]').click()`);
   assert(await evaluate(`traceState.mode==='pan'&&document.getElementById('trace-charts').classList.contains('pan')&&document.querySelector('[data-trace-mode="pan"]').getAttribute('aria-pressed')==='true'`));
   await drag(.5,.25,area.y);
   close(await view(),[shown[0]+span/4,shown[1]+span/4]);
   shown=await view();
   await mouse('mousePressed',at(.5),area.y,{modifiers:8});await mouse('mouseMoved',at(.75),area.y,{modifiers:8});await mouse('mouseReleased',at(.75),area.y,{modifiers:8});
   close(await view(),[shown[0]+span/2,shown[0]+span*3/4]);
   shown=await view();span=shown[1]-shown[0];
   const middle=(shown[0]+shown[1])/2/area.full;
   await drag(middle,middle-.1,area.strip);
   close(await view(),[shown[0]-area.full/10,shown[1]-area.full/10]);
   await mouse('mousePressed',at(.9),area.strip);await mouse('mouseReleased',at(.9),area.strip);
   close(await view(),[area.full*.9-span/2,area.full*.9+span/2]);
   await evaluate(`document.querySelector('[data-trace-mode="zoom"]').click()`);
   await mouse('mousePressed',at(.5),area.y);await mouse('mouseReleased',at(.5),area.y);
   await mouse('mousePressed',at(.5),area.y,{clickCount:2});await mouse('mouseReleased',at(.5),area.y,{clickCount:2});
   assert(await evaluate(`traceState.view===null&&traceState.mode==='zoom'&&traceState.drag===null`));
   // Two fingers each keep the moment of the trace they touched, so spreading them zooms in.
   await call('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:2});
   const touch=(type,points)=>call('Input.dispatchTouchEvent',{type,touchPoints:points.map(([share,id])=>({x:at(share),y:area.y,id}))});
   await touch('touchStart',[[.4,1]]);await touch('touchStart',[[.4,1],[.6,2]]);await touch('touchMove',[[.3,1],[.7,2]]);await touch('touchEnd',[]);
   close(await view(),[area.full/4,area.full*3/4]);
   await call('Emulation.setTouchEmulationEnabled',{enabled:false});
   await evaluate(`showTrace('D-HT-L01',true)`);
   await call('Emulation.clearDeviceMetricsOverride');
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
