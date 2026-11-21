/** Browser QA for the standalone introduction; uses local Chrome and Playwright. */
import {createRequire} from 'node:module';
import {readFile, mkdir, access, writeFile} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const require=createRequire(import.meta.url);
const {chromium}=require(process.env.FRAUDML_PLAYWRIGHT_MODULE || 'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html=path.join(root,'presentation/project-intro.html');
const out=path.join(root,'results/presentation_qa_20261121');
await mkdir(out,{recursive:true});
const summary=JSON.parse(await readFile(path.join(root,'evidence/executed_summary.json'),'utf8'));
const sensitivity=JSON.parse(await readFile(path.join(root,'evidence/supplement_20261121_01/sensitivity.json'),'utf8'));
const omission=sensitivity.periods.test.omissions.find(r=>JSON.stringify(r.omitted_days)==='[171,172]');
const browser=await chromium.launch({headless:true,...(process.env.FRAUDML_CHROME_PATH?{executablePath:process.env.FRAUDML_CHROME_PATH}:{})});
const errors=[], network=[], layouts=[], screenshots=[];
let resultScenarios=0, toyScenarios=0, dailyScenarios=0, linkChecks=0;
try {
 const context=await browser.newContext({viewport:{width:1440,height:1000},colorScheme:'light',reducedMotion:'reduce'});
 await context.route(/^https?:/,r=>{network.push(r.request().url());return r.abort();});
 const page=await context.newPage();
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto(pathToFileURL(html).href);
 await page.waitForSelector('#result-bars .barrow');
 const embedded=await page.locator('#study-data').textContent();
 assert.deepEqual(JSON.parse(embedded).metrics,summary.metrics,'embedded data differs from evidence');
 assert.deepEqual(JSON.parse(embedded).omission,omission);
 const models=['LR0','LR1','LR2','LGB-H0','LGB-H1','LGB-H2','LGB-H3'];
 for(const period of ['dev_F1','dev_F2','validation','test']) {
  await page.selectOption('#period-select',period);
  for(const rate of ['0.001','0.005','0.01','0.02']) {
   await page.locator(`[data-rate="${rate}"]`).click();
   const bars=await page.locator('#result-bars .barvalue b').allTextContents();
   for(let i=0;i<models.length;i++) {
    const r=summary.metrics[period][models[i]].period_rates[rate];
    assert.equal(bars[i],`${r.hits.toLocaleString('zh-CN')} / ${r.reviews.toLocaleString('zh-CN')}`);
   }
   assert.equal(await page.locator('#result-table tr').count(),9);
   const table=await page.locator('#result-table tr').allTextContents();
   assert.ok(table[8].includes(summary.metrics[period].LR0.period_rates[rate].random_expected_hits.toFixed(2)));
   resultScenarios++;
  }
 }
 // Explicit known examples catch a wrong result field or a stale default.
 await page.selectOption('#period-select','test');
 await page.locator('[data-rate="0.005"]').click();
 assert.deepEqual(await page.locator('#result-bars .barvalue b').allTextContents(),['0 / 443','376 / 443','383 / 443','412 / 443','405 / 443','406 / 443','405 / 443']);
 const toyModelHits=[1,1,2,2,2,3,3,3],toyAmountHits=[0,0,0,0,0,0,0,0];
 for(const mode of ['model','amount']) {
  await page.locator(`#toy-${mode}`).click();
  for(let k=1;k<=8;k++) {
   await page.locator('#toy-k').evaluate((el,k)=>{el.value=k;el.dispatchEvent(new Event('input',{bubbles:true}));},k);
   const hits=(mode==='model'?toyModelHits:toyAmountHits)[k-1];
   assert.equal(await page.locator('#toy-queue .selected').count(),k);
   assert.equal(await page.locator('#toy-hits').textContent(),`${hits} / ${k}`);
   assert.equal(await page.locator('#toy-precision').textContent(),(hits/k*100).toFixed(2)+'%');
   assert.equal(await page.locator('#toy-recall').textContent(),(hits/4*100).toFixed(2)+'%');
   toyScenarios++;
  }
 }
 for(const omit of [false,true]) {
  await page.locator(omit?'#daily-omit':'#daily-full').click();
  const rows=await page.locator('#daily-table tr').evaluateAll(rs=>rs.map(r=>[...r.cells].map(c=>c.textContent)));
  for(const [i,m] of ['LR0','LR2','LGB-H0','LGB-H3'].entries()) {
   const r=omit?omission.models[m]:summary.metrics.test[m].daily.fixed_16;
   const dh=omit?r.daily_hits:r.hits,gh=omit?r.matched_global_hits:r.matched_global.hits,d=dh-gh;
   assert.deepEqual(rows[i],[m,String(dh),String(gh),(d>0?'+':'')+d]);
  }
  assert.ok((await page.locator('#daily-context').textContent()).includes(omit?'464':'496'));
  dailyScenarios++;
 }
 for(const href of await page.locator('a[href]').evaluateAll(as=>as.map(a=>a.getAttribute('href')))) {
  if(href.startsWith('#'))assert.equal(await page.locator(href).count(),1);
  else {assert.ok(!/^[a-z]+:/i.test(href),'unexpected external link');await access(path.resolve(path.dirname(html),decodeURIComponent(href)));}
  linkChecks++;
 }
 await page.locator('#toy-model').click();
 await page.locator('#toy-k').evaluate(el=>{el.value=3;el.dispatchEvent(new Event('input',{bubbles:true}));});
 await page.locator('#daily-full').click();
 async function capture(name,locator,isSection=false){const file=path.join(out,name+'.png');await locator.screenshot({path:file,animations:'disabled',...(isSection?{style:'header { visibility:hidden !important; }'}:{})});screenshots.push(path.relative(root,file));}
 for(const [width,height,scheme] of [[1440,1000,'light'],[390,844,'light'],[320,844,'light'],[1440,1000,'dark']]) {
  await page.setViewportSize({width,height});await page.emulateMedia({colorScheme:scheme,reducedMotion:'reduce'});
  const layout=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));
  assert.ok(layout.scrollWidth<=layout.width+1,`page overflows at ${width}: ${layout.scrollWidth}`);
  layouts.push({...layout,scheme});
  await page.evaluate(()=>scrollTo(0,0));
  await page.waitForFunction(()=>scrollY===0);
  await page.waitForTimeout(150);
  await capture(`intro-${width}-${scheme}-hero`,page);
  await capture(`intro-${width}-${scheme}-results`,page.locator('#results .lab'),true);
  if(width===1440&&scheme==='light'||width===390) {
   await capture(`intro-${width}-${scheme}-toy`,page.locator('#idea .lab'),true);
   await capture(`intro-${width}-${scheme}-daily`,page.locator('#daily .lab'),true);
  }
 }
 assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
 const report={status:'passed',created_at_utc:new Date().toISOString(),browser:browser.version(),html_sha256:createHash('sha256').update(await readFile(html)).digest('hex'),result_scenarios:resultScenarios,toy_scenarios:toyScenarios,daily_scenarios:dailyScenarios,link_checks:linkChecks,layouts,external_network_requests:network,console_or_page_errors:errors,screenshots};
 await writeFile(path.join(out,'browser-verification.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report,null,2));
} finally {await browser.close();}
