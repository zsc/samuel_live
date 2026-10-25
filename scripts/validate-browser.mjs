import {gzipSync} from 'node:zlib';
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
    const data=await readFile(file);res.setHeader('Content-Type',mime[extname(file)]??'application/octet-stream');if(process.env.GZIP && file.endsWith('.onnx')){const zipped=gzipSync(data);res.setHeader('Content-Encoding','gzip');res.setHeader('Content-Length',zipped.length);res.end(zipped);}else res.end(data);
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
  await ready(process.env.WASM?'&wasm=1':'');report.initial=await telemetry();console.log('MODEL READY',JSON.stringify(report.initial));
  await page.locator('#metrics-panel').evaluate(e=>e.open=true);
  await page.click('#selftest-button');
  await page.waitForFunction(()=>window.__samuel.telemetry.selfTest,{},{timeout:180000});
  report.numerical=(await telemetry()).selfTest;console.log('NUMERICAL',JSON.stringify(report.numerical));
  if(process.env.QUICK){ await page.screenshot({path:'docs/browser-ready.png',fullPage:true}); }
  else {
    assert.equal(report.numerical.passed,true,'numerical reference failed');
    await page.click('#fixture-button');
    await page.waitForFunction(()=>document.body.dataset.state==='recording',{},{timeout:15000});
    for(let i=0;i<3;i++){await page.waitForTimeout(20000);const t=await telemetry();console.log('LIVE',i+1,JSON.stringify({frames:t.lastFrame,stats:t.stats,displayLatencyP95:t.displayLatencyP95,state:t.state}));}
    await page.waitForFunction(()=>document.body.dataset.state==='playable',{},{timeout:30000});
    report.live=await telemetry();report.recording=await page.evaluate(()=>{const r=window.__samuel.recording;return {frames:r.frameCount,samples:r.validModelSamples,sourceSamples:r.originalPcm.length,sourceRate:r.sourceSampleRate,duration:r.durationSeconds,config:r.processingConfig};});
    assert.equal(report.recording.duration,60);assert.equal(report.recording.samples,2646000);assert.equal(report.recording.frames,5168);assert.equal(report.live.captureTracks.length,0);assert.ok(report.live.liveMessages>1000);
    await page.screenshot({path:'docs/browser-recorded.png',fullPage:true});
    await page.evaluate(()=>{
      const s=window.__samuel.synth, ctx=s.context;
      const tap=new AudioWorkletNode(ctx,"samuel-mic-capture",{processorOptions:{sessionId:999,chunkFrames:4096}});
      const silent=ctx.createGain();silent.gain.value=0;s.masterGain.connect(tap);tap.connect(silent).connect(ctx.destination);
      window.__tap={node:tap,parts:[],silent};
      tap.port.onmessage=e=>{if(e.data.type==='chunk')window.__tap.parts.push(new Float32Array(e.data.pcm));else if(e.data.type==='eos')window.__tap.done=true;};
    });
    await page.click('#play-button');await page.waitForTimeout(31000);
    const positionAfter31s=await page.evaluate(()=>window.__samuel.player.position());
    await page.evaluate(()=>window.__tap.node.port.postMessage({type:'stop'}));
    await page.waitForFunction(()=>window.__tap.done);
    const audio=await page.evaluate(()=>{
      const t=window.__tap;window.__samuel.synth.masterGain.disconnect(t.node);t.node.disconnect();t.silent.disconnect();
      return {rate:window.__samuel.synth.context.sampleRate,pcm:t.parts.flatMap(p=>Array.from(p))};
    });
    const pcm=Float32Array.from(audio.pcm);let energy=0,peak=0,nonfinite=0;
    for(const v of pcm){energy+=v*v;peak=Math.max(peak,Math.abs(v));if(!Number.isFinite(v))nonfinite++;}
    report.audio={seconds:pcm.length/audio.rate,rms:Math.sqrt(energy/pcm.length),peak,nonfinite};
    assert.equal(nonfinite,0);assert.ok(report.audio.rms>.001);assert.ok(peak<1);
    const wav=Buffer.alloc(44+pcm.length*2);wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(audio.rate,24);wav.writeUInt32LE(audio.rate*2,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(pcm.length*2,40);
    pcm.forEach((v,i)=>wav.writeInt16LE(Math.round(Math.max(-1,Math.min(1,v))*32767),44+i*2));
    await writeFile('docs/samuel-browser-replay.wav',wav);

    report.playback=await telemetry();report.playback.position=positionAfter31s;
    assert.ok(report.playback.position>30 && report.playback.position<33);assert.ok(report.playback.drawCount>100);
    await page.click('#pause-button');const paused=await page.evaluate(()=>window.__samuel.player.position());await page.waitForTimeout(250);assert.equal(await page.evaluate(()=>window.__samuel.player.position()),paused);
    await page.click('#play-button');await page.waitForTimeout(500);
    await page.locator('#progress').fill('700');await page.locator('#progress').dispatchEvent('input');
    assert.equal((await telemetry()).state,'paused');assert.ok(Math.abs(await page.evaluate(()=>window.__samuel.player.position())-42)<.01);
    await page.click('#play-button');await page.waitForTimeout(500);
    await page.check('#original-toggle');await page.waitForTimeout(500);
    await page.uncheck('#original-toggle');await page.waitForTimeout(500);
    await page.click('#restart-button');await page.waitForTimeout(300);assert.ok(await page.evaluate(()=>window.__samuel.player.position())<1);
    await page.click('#pause-button');
    report.transport='pause/resume/seek/mode/restart passed';
    await page.click('#fixture-button');await page.waitForTimeout(1200);await page.click('#stop-button');
    await page.waitForFunction(()=>document.body.dataset.state==='playable');
    report.rerecord=await page.evaluate(()=>{const r=window.__samuel.recording;return {frames:r.frameCount,samples:r.validModelSamples,duration:r.durationSeconds};});
    assert.equal(report.rerecord.frames,Math.ceil(report.rerecord.samples/512));assert.ok(report.rerecord.duration<3);
  }
}catch(e){report.failure=String(e);console.error(e);process.exitCode=1;}
finally {
  report.errors=errors;report.requests=requests;
  await mkdir('docs',{recursive:true});
  await writeFile(`docs/browser-${process.env.WASM?'wasm':process.env.QUICK?'webgpu-numerical':'webgpu'}.json`,JSON.stringify(report,null,2));
  await writeFile(`docs/browser-${process.env.WASM?'wasm':process.env.QUICK?'webgpu-numerical':'webgpu'}.log`,logs.join('\n'));
  await browser.close();server.close();
}
