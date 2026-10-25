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
const browser=await chromium.launch({channel:'chrome',headless:process.env.HEADED!=='1',args:['--autoplay-policy=no-user-gesture-required','--disable-background-timer-throttling','--disable-renderer-backgrounding','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--use-file-for-fake-audio-capture='+resolve('public/fixtures/samuel-sample-48k.wav')]});
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
  await ready();
  await page.click('#record-button');
  await page.waitForFunction(()=>document.body.dataset.state==='recording');
  await page.waitForTimeout(3500);
  const live=await telemetry();assert.ok(live.liveMessages>20);
  await page.click('#stop-button');await page.waitForFunction(()=>document.body.dataset.state==='playable');
  report.fakeMicrophone=await page.evaluate(()=>{const r=window.__samuel.recording;return {...window.__samuel.telemetry,sourceKind:r.sourceKind,samples:r.validModelSamples,frames:r.frameCount,sourceSamples:r.originalPcm.length,sourceRate:r.sourceSampleRate,duration:r.durationSeconds};});
  assert.equal(report.fakeMicrophone.sourceKind,'microphone');assert.equal(report.fakeMicrophone.captureTracks.length,0);
  assert.equal(report.fakeMicrophone.frames,Math.ceil(report.fakeMicrophone.samples/512));
  await page.click('#record-button');await page.waitForTimeout(500);
  await page.evaluate(()=>window.__samuel.revokeInput());
  await page.waitForFunction(()=>document.body.dataset.state==='playable');
  report.revoked='capture ended and existing samples finalized';
  await page.click('#play-button');await page.waitForTimeout(200);await page.click('#pause-button');
  await page.evaluate(async()=>{const p=window.__samuel.player;await Promise.all([p.play(),p.seek(.1),p.play(),p.restart()]);});
  await page.waitForTimeout(300);report.rapidControls=await telemetry();assert.equal(errors.length,0);
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:'docs/browser-mobile-layout.png',fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth),false);
  assert.equal(await page.evaluate(()=>{const a=document.querySelector('#tract-host').getBoundingClientRect();const b=document.querySelector('#tract-scaler').getBoundingClientRect();return b.left>=a.left-.1 && b.right<=a.right+.1;}),true,'scaled tract stays fully inside viewport');
  report.layout='390px viewport no horizontal overflow; desktop Chrome, not mobile hardware';
  console.log('MICROPHONE API / controlled file source passed');
} catch(e){report.failure=String(e);console.error(e);process.exitCode=1;}
finally {report.errors=errors;await writeFile('docs/browser-microphone.json',JSON.stringify(report,null,2));await browser.close();server.close();}
