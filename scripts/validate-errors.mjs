import {chromium} from '@playwright/test';
import {createServer} from 'node:http';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,extname} from 'node:path';
import assert from 'node:assert/strict';
const mime={'.js':'application/javascript','.mjs':'application/javascript','.wasm':'application/wasm','.json':'application/json','.html':'text/html','.css':'text/css','.wav':'audio/wav'};
const server=createServer(async(req,res)=>{
  try {const u=new URL(req.url,'http://localhost'); if(!u.pathname.startsWith('/samuel-live/')) {res.writeHead(404).end();return;}
    const file=resolve('dist',decodeURIComponent(u.pathname.slice('/samuel-live/'.length))||'index.html');
    if(!file.startsWith(resolve('dist')+'/')) throw Error('path');
    const data=await readFile(file);res.setHeader('Content-Type',mime[extname(file)]??'application/octet-stream');res.end(data);
  }catch {res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const base=`http://127.0.0.1:${server.address().port}/samuel-live/`;
const browser=await chromium.launch({channel:'chrome',headless:process.env.HEADED!=='1',args:['--autoplay-policy=no-user-gesture-required','--disable-background-timer-throttling','--disable-renderer-backgrounding']});
const logs=[],errors=[],requests=[];const report={date:new Date().toISOString(),browser:browser.version(),subpath:base};
const page=await browser.newPage({viewport:{width:1280,height:1000}});
page.on('console',m=>logs.push(`${m.type()}: ${m.text()}`));page.on('pageerror',e=>errors.push(e.message));
page.on('response',r=>{if(r.status()>=400 && !r.url().endsWith('favicon.ico'))errors.push(`${r.status()} ${r.url()}`);requests.push({url:new URL(r.url()).pathname,status:r.status(),mime:r.headers()['content-type']});});
const telemetry=()=>page.evaluate(()=>window.__samuel.telemetry);
async function ready(suffix=''){
  await page.goto(base+'?validate=1'+suffix);
  await page.waitForFunction(()=>['ready','error'].includes(document.body.dataset.state),{},{timeout:90000});
  assert.equal((await telemetry()).state,'ready',await page.locator('#status-detail').innerText());
}

try {
  await ready('&delay=200');
  await page.click('#fixture-button');
  await page.waitForFunction(()=>window.__samuel.telemetry.stopReason,{},{timeout:15000});
  await page.waitForFunction(()=>document.body.dataset.state==='playable',{},{timeout:30000});
  report.backpressure=await page.evaluate(()=>{const r=window.__samuel.recording;return {...window.__samuel.telemetry,frames:r.frameCount,samples:r.validModelSamples,sourceSamples:r.originalPcm.length,sourceRate:r.sourceSampleRate,duration:r.durationSeconds};});
  assert.match(report.backpressure.stopReason,/处理落后/);
  assert.equal(report.backpressure.frames,Math.ceil(report.backpressure.samples/512));
  assert.equal(report.backpressure.samples,Math.ceil(report.backpressure.sourceSamples*44100/report.backpressure.sourceRate));
  assert.ok(report.backpressure.duration<10);
  console.log('BACKPRESSURE passed',report.backpressure.duration);
  await ready('&rate=44100');
  await page.click('#fixture-button');await page.waitForTimeout(1600);await page.click('#stop-button');
  await page.waitForFunction(()=>document.body.dataset.state==='playable');
  report.rate44100=await page.evaluate(()=>{const r=window.__samuel.recording;return {sourceRate:r.sourceSampleRate,samples:r.validModelSamples,sourceSamples:r.originalPcm.length,frames:r.frameCount,duration:r.durationSeconds};});
  assert.equal(report.rate44100.sourceRate,44100);assert.equal(report.rate44100.sourceSamples,report.rate44100.samples);
  await page.evaluate(()=>Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async()=>{throw new DOMException('simulated permission denial','NotAllowedError');}}));
  await page.click('#record-button');await page.waitForFunction(()=>window.__samuel.telemetry.error);
  report.denied=await page.locator('#status-detail').innerText();assert.match(report.denied,/denial/);
  assert.equal(await page.locator('#record-button').isEnabled(),true);
  await page.evaluate(()=>window.__samuel.loseDevice());
  await page.waitForFunction(()=>document.querySelector('#status-detail').textContent.includes('GPU device lost'),{},{timeout:15000});
  report.deviceLost=await page.locator('#status-detail').innerText();
  assert.equal(await page.locator('#play-button').isEnabled(),true);
  await page.click('#play-button');await page.waitForTimeout(250);await page.click('#pause-button');
  report.playbackAfterDeviceLoss='passed';
  await page.route('**/models/*.onnx',route=>route.fulfill({status:503,body:'simulated unavailable'}));
  await page.goto(base+'?validate=1');await page.waitForFunction(()=>document.body.dataset.state==='error');
  report.downloadFailure=await page.locator('#status-detail').innerText();assert.match(report.downloadFailure,/503/);
  await page.unroute('**/models/*.onnx');await page.click('#retry-button');await page.waitForFunction(()=>document.body.dataset.state==='ready');
  report.retry='passed';
  await page.route('**/assets/inference.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:'Object.defineProperty(navigator, \"gpu\", {value:undefined});\n'+await response.text()});});
  await ready();report.noGpu=await telemetry();assert.equal(report.noGpu.backend.actual,'wasm');
  console.log('ERROR RECOVERY passed');
} catch(e) {report.failure=String(e);console.error(e);process.exitCode=1;}
finally {
  report.errors=errors;await writeFile('docs/browser-errors.json',JSON.stringify(report,null,2));
  await writeFile('docs/browser-errors.log',logs.join('\n'));await browser.close();server.close();
}
