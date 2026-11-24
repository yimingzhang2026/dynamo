// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Capture the figures of the Planner Gym README from the saved pages, through Chrome DevTools.
//
//   CHROME_ENDPOINT=http://127.0.0.1:9222 node capture_figures.mjs
//
// Each figure is one region of one page. Numbered marks are drawn on the page before the capture;
// the README explains every number. Run it again after publishing results, so the figures show
// the saved numbers.
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';

const root=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(root,'../../docs/images/leaderboard');
const endpoint=process.env.CHROME_ENDPOINT || 'http://127.0.0.1:9247';
// The trace the README follows through the tabs, and the period its charts show.
const trace='A-HT-S03',period=[8*3600,14*3600],hover={panel:'ready_decode',share:.62};

// page: the saved page. prepare: script run first. from, to: the region runs from the top of one
// element to the bottom of the other. frame: the element giving the width (the page content by
// default). marks: numbered in order; each sits beside its element's content, at the given
// fractions of its box plus an offset in pixels.
const figures=[
 {name:'overview',page:'index.html',from:'.topbar',to:'.overview-grid',marks:[
  {on:'.nav',x:0,dx:-18},{on:'.hero .tag',x:0,dx:-18},{on:'.overview-card .winner',x:1,dx:18},
  {on:'.overview-card th:nth-child(2)',x:1,dx:16},{on:'.overview-card .primary-link',x:1,dx:20,box:true}]},
 {name:'topology-ranking',page:'aggregated.html',from:'.filterbar',to:'#core-ranking',marks:[
  {on:'.segmented',x:1,dx:20,box:true},{on:'#view-summary .stat:nth-child(2) .stat-value',x:1,dx:18},
  {on:'#core-ranking th:nth-child(3)',x:1,dx:16},{on:'#core-ranking th:nth-child(4)',x:1,dx:16},
  {on:'#core-ranking th:nth-child(5)',x:1,dx:16}]},
 {name:'topology-cases',page:'aggregated.html',from:'section:has(#case-rows)',to:'#case-rows tr:nth-child(3)',marks:[
  {on:'#search',x:1,dx:20,box:true},{on:'#case-rows tr:nth-child(1) .experiment-case',x:1,dx:16},
  {on:`#case-rows [data-case="${trace}"]`,x:1,dx:16},{on:'#case-rows tr:nth-child(3) .best',x:1,dx:16}]},
 {name:'latency',page:'latency.html',from:'#latency-agg',to:'#latency-agg tbody tr:nth-child(3)',marks:[
  {on:`#latency-agg th[data-trace="${trace}"] .case-button`,x:1,dx:16},
  {on:`#latency-agg td[data-case="${trace}"][data-planner="mpc"] .latency-health`,x:0,dx:-22},
  {on:`#latency-agg td[data-case="${trace}"][data-planner="mpc"] [data-metric="itl"] .latency-label`,x:1,dx:15},
  {on:`#latency-agg td[data-case="${trace}"][data-planner="keda"] .diagnose-button`,x:1,dx:16}]},
 {name:'diagnosis',page:'latency.html',prepare:`document.querySelector('[data-diagnose-case="${trace}"][data-diagnose-planner="keda"]').click()`,
  from:'#case-dialog',to:'#case-dialog',frame:'#case-dialog',marks:[]},
 {name:'slo-rankings',page:'slo-rankings.html',from:'.hero',to:'.overview-note',marks:[
  {on:'.hero .tag',x:0,dx:-18},{on:'.overview-card th:nth-child(2)',x:1,dx:16},
  {on:'.overview-card .table-scroll + p',x:1,dx:16}]},
 {name:'trace-table',page:'traces.html#'+trace,from:'.trace-controls',to:'#trace-leaderboard',marks:[
  {on:'#trace-next',x:1,dx:20,box:true},{on:'#trace-specs div:nth-child(8)',x:1,dx:20},
  {on:'#trace-leaderboard th:nth-child(3)',x:.5,y:0,dy:-2,box:true},{on:'#trace-leaderboard th:nth-child(8)',x:.5,y:0,dy:-2,box:true},
  {on:'#trace-leaderboard th:nth-child(13)',x:.5,y:0,dy:-2,box:true},{on:'#trace-leaderboard th:nth-child(15)',x:.5,y:0,dy:-2,box:true}]},
 {name:'trace-behavior',page:'traces.html#'+trace,
  prepare:`traceSetView(${period[0]},${period[1]});(()=>{const c=document.querySelector('[data-trace-panel="${hover.panel}"] canvas'),b=c.getBoundingClientRect();c.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,pointerType:'mouse',clientX:b.left+b.width*${hover.share},clientY:b.top+b.height*.45}));})()`,
  from:'#trace-behavior',to:'#trace-behavior',marks:[
  {on:'#trace-legend label:last-child',x:1,dx:18},{on:'.trace-zoom .segmented',x:0,dx:-13,box:true},
  {on:'#trace-fit',x:0,dx:-17,box:true},{on:'#trace-nav',x:0,dx:-13,box:true},
  {on:'[data-trace-panel="ttft_ms"] .trace-panel-title > span',x:0,dx:-17},{on:'#trace-tip',x:1,y:0,dx:2,dy:-2,box:true}]},
];

// Runs in the page: draws the marks and returns the region to capture.
const place=({from,to,frame,marks})=>{
 const box=(selector,whole)=>{
  const element=document.querySelector(selector);
  if(!element)throw new Error('No element for '+selector);
  if(!whole){const range=document.createRange();range.selectNodeContents(element);const text=range.getBoundingClientRect();if(text.width&&text.height)return text;}
  return element.getBoundingClientRect();
 };
 marks.forEach((mark,index)=>{
  const at=box(mark.on,mark.box),badge=document.createElement('b');
  badge.textContent=index+1;
  badge.style.cssText=`position:absolute;z-index:50;left:${at.left+scrollX+at.width*(mark.x??0)+(mark.dx??0)-12}px;top:${at.top+scrollY+at.height*(mark.y??.5)+(mark.dy??0)-12}px;width:24px;height:24px;border-radius:50%;background:#c2255c;color:#fff;font:700 14px/24px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;text-align:center;box-shadow:0 0 0 2px #fff,0 2px 6px #0005;pointer-events:none`;
  document.body.appendChild(badge);
 });
 // The region stays inside the page: a capture reaching outside it is not taken at the device scale.
 const first=box(from,true),last=box(to,true),width=box(frame||'#app',true),margin=frame?18:14;
 const x=Math.max(0,width.left-margin),y=Math.max(0,first.top+scrollY-margin);
 return {x,y,width:Math.min(innerWidth,width.right+margin)-x,height:last.bottom+scrollY+margin-y,scale:1};
};

const target=await (await fetch(endpoint+'/json/new?about:blank',{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve=>socket.addEventListener('open',resolve,{once:true}));
let nextId=0;
const pending=new Map(),exceptions=[];
socket.addEventListener('message',event=>{
 const value=JSON.parse(event.data);
 if(value.method==='Runtime.exceptionThrown')exceptions.push(value.params.exceptionDetails);
 if(value.id&&pending.has(value.id)){
  const {resolve,reject}=pending.get(value.id);pending.delete(value.id);
  value.error?reject(new Error(JSON.stringify(value.error))):resolve(value.result);
 }
});
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
async function evaluate(expression){
 const value=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
 assert(!value.exceptionDetails,JSON.stringify(value.exceptionDetails));return value.result.value;
}
const settle=()=>evaluate(`new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>setTimeout(resolve,150))))`);
// Twice the pixels of the layout, so text stays sharp at the width of a README.
const view=height=>call('Emulation.setDeviceMetricsOverride',{width:1280,height,deviceScaleFactor:2,mobile:false});
await call('Runtime.enable');await call('Page.enable');
await mkdir(out,{recursive:true});
try{
 for(const figure of figures){
  await view(900);
  await call('Page.navigate',{url:'about:blank'});
  await call('Page.navigate',{url:pathToFileURL(path.join(root,figure.page.split('#')[0])).href+(figure.page.includes('#')?'#'+figure.page.split('#')[1]:'')});
  for(let i=0;;i++){
   if(await evaluate(`document.readyState==='complete'&&typeof report!=='undefined'&&document.getElementById('app').textContent.length>100`))break;
   assert(i<100,'Page did not render: '+figure.page);
   await new Promise(resolve=>setTimeout(resolve,50));
  }
  // The view grows to the whole page, so the region is captured exactly as it is laid out.
  await view(Math.max(900,Math.ceil(await evaluate(`document.documentElement.scrollHeight`))));
  await settle();
  if(figure.prepare)await evaluate(figure.prepare);
  await settle();
  const clip=await evaluate(`(${place})(${JSON.stringify(figure)})`);
  await settle();
  const shot=await call('Page.captureScreenshot',{format:'png',clip});
  await writeFile(path.join(out,figure.name+'.png'),Buffer.from(shot.data,'base64'));
  console.log(`${figure.name}.png ${Math.round(clip.width)} x ${Math.round(clip.height)}`);
 }
 assert.equal(exceptions.length,0,JSON.stringify(exceptions));
}finally{await call('Page.close').catch(()=>{});socket.close();}
