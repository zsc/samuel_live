import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const manifest=JSON.parse(readFileSync('public/models/samuel-manifest.json','utf8'));
const sha=p=>createHash('sha256').update(readFileSync(p)).digest('hex');
if(sha(`public/models/${manifest.onnxFile}`)!==manifest.onnxSha256)throw Error('Model SHA-256 mismatch');
if(sha('public/models/config.json')!==manifest.browserConfigSha256)throw Error('Browser config SHA-256 mismatch');
const provenance=JSON.parse(readFileSync('public/fixtures/provenance.json','utf8'));
for(const [name,hash] of Object.entries(provenance.files))if(sha('public/fixtures/'+name)!==hash)throw Error('Fixture SHA-256 mismatch: '+name);
const fixtures=JSON.parse(readFileSync('public/fixtures/cases.json','utf8'));
for(const f of fixtures){
  if(readFileSync(`public/fixtures/${f.pcmFile}`).length!==f.sampleCount*4)throw Error(`PCM mismatch ${f.name}`);
  if(readFileSync(`public/fixtures/${f.logitsFile}`).length!==f.frameCount*7*32*4)throw Error(`Logits mismatch ${f.name}`);
}
console.log(`Verified pinned model ${manifest.modelRevision}, ${fixtures.length} reference cases`);
