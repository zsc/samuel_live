import {describe,it,expect} from 'vitest';
import {StreamingSincResampler} from '../src/audio/resampler';
import {CausalRmsNormalizer} from '../src/audio/rms';
import {CausalPitchTracker} from '../src/audio/pitch';
const concat = (chunks:Float32Array[]) => Float32Array.from(chunks.flatMap(c=>Array.from(c)));
describe('continuous audio preprocessing',()=>{
  for(const rate of [44100,48000]) it(`${rate}: exact duration, phase, pitch and idempotent tail`,()=>{
    const source=Float32Array.from({length:rate+137},(_,i)=>.1*Math.sin(2*Math.PI*220*i/rate));
    const whole=new StreamingSincResampler(rate,44100);
    const expected=concat([whole.push(source),whole.flush()]);
    const split=new StreamingSincResampler(rate,44100); const chunks=[];
    for(let pos=0;pos<source.length;) {const n=[128,1024,77,2048][pos%4];chunks.push(split.push(source.slice(pos,pos+n)));pos+=n;}
    chunks.push(split.flush());const actual=concat(chunks);
    expect(actual.length).toBe(Math.ceil(source.length*44100/rate));
    expect(split.flush().length).toBe(0);expect(actual).toEqual(expected);
    const tracker=new CausalPitchTracker();let result;
    for(let f=0;f<Math.floor(actual.length/512);f++) result=tracker.frame(actual,f);
    expect(result?.voiced).toBe(true);expect(Math.abs(result!.f0-220)).toBeLessThan(.2);
  });
  it('RMS is causal, chunk-invariant and bounded on silence',()=>{
    const input=Float32Array.from({length:10000},(_,i)=>i<500?0:.1*Math.sin(i*.02));
    const a=new CausalRmsNormalizer(), b=new CausalRmsNormalizer();
    const whole=a.process(input);const split=concat([b.process(input.slice(0,777)),b.process(input.slice(777))]);
    expect(split).toEqual(whole);expect(a.lastGain).toBeLessThanOrEqual(20);
    expect(Array.from(whole.slice(0,500)).every(x=>x===0)).toBe(true);
    const silence=new CausalPitchTracker(); const pcm=new Float32Array(4096);
    for(let f=0;f<8;f++){ const result=silence.frame(pcm,f)!;expect(result.voiced).toBe(false);expect(result.f0).toBe(285); }
  });
  it('pitch ignores future samples and resets independently',()=>{
    const pcm=Float32Array.from({length:8192},(_,i)=>.1*Math.sin(2*Math.PI*180*i/44100));
    const a=new CausalPitchTracker(),b=new CausalPitchTracker();
    for(let f=0;f<16;f++) expect(a.frame(pcm,f)).toEqual(b.frame(pcm.slice(0,(f+1)*512),f));
    a.reset();expect(a.frame(pcm,0)?.f0).toBe(285);
  });
});

import {readFileSync} from 'node:fs';
it('causal YIN matches pinned upstream on real speech',()=>{
  const ref=JSON.parse(readFileSync('public/fixtures/pitch-reference.json','utf8')) as {f0:number[];voiced:boolean[]};
  const bytes=readFileSync('public/fixtures/speech.f32');
  const pcm=new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  const tracker=new CausalPitchTracker();
  for(let f=0;f<ref.f0.length;f++){
    const value=tracker.frame(pcm,f)!;
    expect(value.voiced,`voiced frame ${f}`).toBe(ref.voiced[f]);
    expect(Math.abs(value.f0-ref.f0[f]),`F0 frame ${f}`).toBeLessThan(.1);
  }
});
