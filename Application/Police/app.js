// Static Police Dashboard wiring
// Loads insights JSONs and renders police-facing summaries

const PATHS = {
  map: '../../JupyterOutputs/Clustering (SpatialHotspots)/cluster_temporal_patterns_map.html',
  executive: '../../JupyterOutputs/Clustering (MultidimensionalClusteringAnalysis)/executive_crime_summary_enriched.json',
  execFallback: '../../JupyterOutputs/Clustering (MultidimensionalClusteringAnalysis)/executive_crime_summary.json',
  patternGlobal: '../../JupyterOutputs/PatternAnalysis/global_insights.txt',
  patternBorough: (b)=>`../../JupyterOutputs/PatternAnalysis/Borough_${b}_insights.txt`,
  patternTime: (t)=>`../../JupyterOutputs/PatternAnalysis/TimeBucket_${t}_insights.txt`
};

// thresholds to filter weak rules
const RULE_FILTER = { minConfidencePct: 60, minLift: 1.05, minSupport: 0.10 };

function parseMetrics(metricLine){
  const rawLine = metricLine ? metricLine.replace(/^\-\s*/, '').trim() : '';
  if(!rawLine){ return { conf: undefined, lift: undefined, supp: undefined, kulc: undefined, score: undefined, raw: '' }; }
  const entries = rawLine.split(',').map(s=>s.trim()).map(part=>{
    const idx = part.indexOf(':');
    if(idx===-1) return [part.toLowerCase(), null];
    const key = part.slice(0, idx).trim().toLowerCase();
    const val = part.slice(idx+1).trim();
    return [key, val];
  });
  const obj = Object.fromEntries(entries);
  const pct = v => v!=null ? parseFloat(String(v).replace('%','')) : undefined;
  const num = v => v!=null ? parseFloat(String(v)) : undefined;
  const conf = obj.confidence!=null ? pct(obj.confidence) : (obj.conf!=null ? pct(obj.conf) : undefined);
  const lift = obj.lift!=null ? num(obj.lift) : undefined;
  const supp = obj.support!=null ? num(obj.support) : (obj.supp!=null ? num(obj.supp) : undefined);
  const kulc = obj.kulc!=null ? num(obj.kulc) : undefined;
  const score = obj.score!=null ? num(obj.score) : undefined;
  const pretty = [];
  if(kulc!=null && !Number.isNaN(kulc)) pretty.push(`Kulc: ${kulc}`);
  if(conf!=null && !Number.isNaN(conf)) pretty.push(`Conf: ${conf.toFixed(2)}%`);
  if(lift!=null && !Number.isNaN(lift)) pretty.push(`Lift: ${lift}`);
  if(supp!=null && !Number.isNaN(supp)) pretty.push(`Supp: ${supp}`);
  if(score!=null && !Number.isNaN(score)) pretty.push(`Score: ${score}`);
  return { conf, lift, supp, kulc, score, raw: pretty.join(', ') };
}

function passesFilter(metrics){
  const { conf, lift, supp } = metrics || {};
  if(Number.isFinite(conf) && conf < RULE_FILTER.minConfidencePct) return false;
  if(Number.isFinite(lift) && lift < RULE_FILTER.minLift) return false;
  if(Number.isFinite(supp) && supp < RULE_FILTER.minSupport) return false;
  return true;
}

function $(sel){ return document.querySelector(sel); }
function el(tag, cls, text){ const n = document.createElement(tag); if(cls) n.className=cls; if(text!==undefined) n.textContent=text; return n; }

async function loadJSON(url){
  try{ const res = await fetch(url, { cache: 'no-store' }); if(!res.ok) throw new Error(`${res.status} ${res.statusText}`); return await res.json(); }
  catch(err){ console.warn('Failed to load', url, err); return null; }
}
async function loadText(url){
  try{ const res = await fetch(url, { cache: 'no-store' }); if(!res.ok) throw new Error(`${res.status} ${res.statusText}`); return await res.text(); }
  catch(err){ console.warn('Failed to load', url, err); return null; }
}

function setLastUpdated(analysisDate){
  const node = $('#last-updated');
  node.textContent = analysisDate ? `Analysis: ${analysisDate}` : 'Analysis: —';
}

function renderKpis(exec){
  const wrap = $('#kpi-cards');
  wrap.innerHTML = '';
  const kpis = [
    { label: 'Crimes analyzed', value: exec?.total_crimes_analyzed ?? '—' },
    { label: 'Patterns identified', value: exec?.patterns_identified ?? '—' },
    { label: 'High-priority patterns', value: exec?.high_priority_patterns ?? '—' },
    { label: '% of high-priority crimes', value: (exec?.high_priority_crime_percentage!=null) ? `${exec.high_priority_crime_percentage}%` : '—' }
  ];
  for(const k of kpis){ const card = el('div','card'); card.append(el('div','label',k.label)); card.append(el('div','value',String(k.value))); wrap.append(card); }
}

function fmtPattern(p){ if(!p) return '—'; const parts = [p.crime_type, p.borough, p.premises, p.time_bucket]; const w = p.is_weekend_mode ? 'Weekend' : 'Weekday'; return `${parts.filter(Boolean).join(' • ')} • ${w}`; }

function renderPriorityPatterns(exec){
  const list = $('#priority-patterns'); list.innerHTML = '';
  const blocks = [
    { title: 'Most concentrated', data: exec?.most_concentrated_pattern, extra: exec?.most_concentrated_pattern?.concentration },
    { title: 'Highest volume', data: exec?.highest_volume_pattern, extra: exec?.highest_volume_pattern?.volume!=null ? `${exec.highest_volume_pattern.volume} cases` : undefined },
  ];
  for(const b of blocks){ if(!b.data) continue; const item = el('div','pattern'); item.append(el('div','title', `${b.title}: ${fmtPattern(b.data)}`)); const meta = []; if(b.data?.suspect_sex_mode) meta.push(`Suspect: ${b.data.suspect_sex_mode} ${b.data.suspect_age_mode??''}`.trim()); if(b.data?.victim_sex_mode) meta.push(`Victim: ${b.data.victim_sex_mode} ${b.data.victim_age_mode??''}`.trim()); if(b.extra) meta.push(String(b.extra)); if(meta.length) item.append(el('div','meta', meta.join(' • '))); list.append(item); }
}

function deriveRecommendations(exec){
  const recos = []; if(!exec) return recos;
  if(exec.immediate_deployment_needed){ recos.push('Deploy targeted patrols in hotspot areas over the next 7 days (afternoon window).'); }
  if(exec.high_priority_patterns>0){ recos.push(`Allocate resources to ${exec.high_priority_patterns} high-priority patterns with 4-hour rotations.`); }
  if(exec.most_concentrated_pattern?.premises){ recos.push(`Engage with stores/managers: strengthen security in ${exec.most_concentrated_pattern.premises.toLowerCase()}.`); }
  if(exec.most_concentrated_pattern?.borough){ recos.push(`Coordinate NYPD precincts in ${exec.most_concentrated_pattern.borough} for targeted prevention.`); }
  return recos;
}

function renderRecommendations(exec){ const ul = $('#recommendations'); ul.innerHTML = ''; const recos = deriveRecommendations(exec); if(recos.length===0){ ul.append(el('li', null, 'No recommendations available.')); return; } for(const r of recos){ ul.append(el('li', null, r)); } }

// ----- Pattern Analysis (modal) -----
const BOROUGHS = ['MANHATTAN','BROOKLYN','BRONX','QUEENS','STATEN_ISLAND'];
const TIMES = ['MORNING','AFTERNOON','EVENING','NIGHT'];
const MODES = ['GLOBAL','BOROUGH','TIME'];
let currentMode = 'GLOBAL';

function toFriendlyLabel(k,v){
  const map = {
    BORO:'Borough',
    TIME_BUCKET:'Time',
    LOC_OF_OCCUR:'Location',
    DIST_BIN:'Distance to POI',
    HAS_POI:'POI proximity',
    LAW_CAT:'Law Category',
    SUSP_SEX:'Suspect Sex',
    SUSP_AGE:'Suspect Age',
    SUSP_RACE:'Suspect Race',
    VIC_SEX:'Victim Sex',
    VIC_AGE:'Victim Age',
    VIC_RACE:'Victim Race'
  };
  const valMap = { INSIDE:'inside', FRONT:'at front', 'WHITE HISPANIC':'White Hispanic', BLACK:'Black', YES:'yes', NO:'no', MORNING:'morning', AFTERNOON:'afternoon', EVENING:'evening', NIGHT:'night' };
  return { key: map[k] || k, val: (valMap[v] || v) };
}

function formatDistanceToPOI(val){
  const s = String(val || '').toLowerCase();
  if(s.startsWith('<')){
    return `within ${s.slice(1)} of POI`;
  }
  if(s.includes('-')){
    return `${String(val).replace('-', '–')} from POI`;
  }
  return `distance to POI ${val}`;
}

function renderRuleItem(it){
  const box = el('div','pattern');
  if(it.outcome){ box.append(el('div','rule-outcome', it.outcome)); }
  if(it.premise){ const w = el('div','rule-when'); w.append(el('span','label','When ')); w.append(document.createTextNode(it.premise)); box.append(w); }
  if(it.meta){ box.append(el('div','rule-metrics', it.meta)); }
  return box;
}

function parseRuleToSentence(rule, metric){
  const m = rule.match(/^IF\s*\((.+)\)\s*THEN\s*\((.+)\)\s*$/i);
  if(!m){ return { outcome: rule, premise: undefined, meta: metric?.replace(/^\-\s*/, '') }; }
  const lhs = m[1]; const rhs = m[2];
  const parseSide = (side)=> side.split(/\)\s*AND\s*\(/i).map(s=>s.replace(/[()]/g,'')).map(pair=>{
    const [k,v] = pair.split('='); const {key,val} = toFriendlyLabel(k?.trim(), v?.trim()); return {key, val};
  });
  const lhsParts = parseSide(lhs);
  const rhsParts = parseSide(rhs);

  const formatCond = ({key,val}, lower=false)=>{
    if(key==='POI proximity') return (String(val).toLowerCase()==='yes') ? 'near a point of interest' : 'no point of interest nearby';
    if(key==='Distance to POI') return formatDistanceToPOI(val);
    return `${lower ? key.toLowerCase() : key} is ${val}`;
  };

  const outcome = rhsParts.map(c=>formatCond(c,false)).join(' and ');
  const premise = lhsParts.map(c=>formatCond(c,true)).join(' and ');
  return { outcome: `Likely ${outcome}`, premise: premise ? premise : undefined };
}

function parseInsightLines(text){
  if(!text) return [];
  const lines = text.split(/\r?\n/).map(l=>l.trim());
  const items = [];
  for(let i=0;i<lines.length;i++){
    const l = lines[i];
    if(!l || l.startsWith('---')) continue;
    if(l.startsWith('IF ')){
      const metricLine = (i+1<lines.length && lines[i+1].startsWith('-')) ? lines[i+1] : '';
      const metrics = parseMetrics(metricLine);
      if(!passesFilter(metrics)) continue;
      const {outcome, premise} = parseRuleToSentence(l, metricLine);
      items.push({outcome, premise, meta: metrics.raw});
    }
  }
  return items;
}

function renderInsightList(containerSel, items, emptyText){
  const cont = $(containerSel); cont.innerHTML = '';
  if(!items || items.length===0){ cont.textContent = emptyText || 'No insights.'; return; }
  for(const it of items.slice(0,10)) cont.append(renderRuleItem(it));
}

function buildChips(containerId, values, onChange, defaultVal){
  const cont = document.getElementById(containerId); cont.innerHTML = '';
  let current = defaultVal;
  values.forEach(v=>{
    const btn = el('button','chip', v.replace('_',' '));
    btn.dataset.value = v;
    if(v===current) btn.classList.add('active');
    btn.addEventListener('click',()=>{
      if(current===v) return; // no-op if same
      current = v;
      cont.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
      btn.classList.add('active');
      onChange(v);
    });
    cont.append(btn);
  });
  // initial callback
  onChange(current);
}

function getActiveChipValue(containerId, fallbackList){
  const cont = document.getElementById(containerId);
  const active = cont?.querySelector('.chip.active');
  return active?.dataset?.value || (Array.isArray(fallbackList) ? fallbackList[0] : undefined);
}

function wireCollapsibles(){
  document.querySelectorAll('.collapsible').forEach(sec=>{
    const header = sec.querySelector('h3');
    header.addEventListener('click', ()=>{ sec.classList.toggle('collapsed'); });
  });
}

function wirePatternModal(){
  const openBtn = $('#open-pattern-modal'); const modal = $('#pattern-modal'); const closeBtn = $('#pattern-modal-close'); const dismiss = $('#pattern-modal-dismiss');
  const open = ()=>{ modal.classList.remove('hidden'); modal.setAttribute('aria-hidden','false'); };
  const close = ()=>{ modal.classList.add('hidden'); modal.setAttribute('aria-hidden','true'); };
  openBtn.addEventListener('click', open); closeBtn.addEventListener('click', close); dismiss.addEventListener('click', close); document.addEventListener('keydown', (e)=>{ if(e.key==='Escape') close(); });
  
  // Build chips for modes
  buildChips('mode-chips', MODES, (m)=>{
    setMode(m);
    // lazy-load per mode
    if(m==='GLOBAL'){
      loadPatternGlobal();
    } else if(m==='BOROUGH'){
    const b = getActiveChipValue('borough-chips', BOROUGHS);
    if(b) loadPatternBorough(b);
    } else if(m==='TIME'){
    const t = getActiveChipValue('time-chips', TIMES);
    if(t) loadPatternTime(t);
    }
  }, 'GLOBAL');

  // Existing chips
  buildChips('borough-chips', BOROUGHS, (b)=>{ if(currentMode==='BOROUGH') loadPatternBorough(b); }, BOROUGHS[0]);
  buildChips('time-chips', TIMES, (t)=>{ if(currentMode==='TIME') loadPatternTime(t); }, TIMES[0]);
  
  wireCollapsibles();
}

async function loadPatternGlobal(){ const txt = await loadText(PATHS.patternGlobal); const items = parseInsightLines(txt); renderInsightList('#pa-global', items, 'No global insights available.'); }
async function loadPatternBorough(b){ const txt = await loadText(PATHS.patternBorough(b)); const items = parseInsightLines(txt); renderInsightList('#pa-borough', items, 'No borough insights available.'); }
async function loadPatternTime(t){ const txt = await loadText(PATHS.patternTime(t)); const items = parseInsightLines(txt); renderInsightList('#pa-time', items, 'No time-bucket insights available.'); }

function wireMapFallback(){ const iframe = document.getElementById('hotspot-map'); const fallback = document.getElementById('map-fallback'); let shown = false; const showFallback = ()=>{ if(shown) return; shown=true; fallback.style.display = 'flex'; }; iframe.addEventListener('error', showFallback); setTimeout(()=>{ try{ void iframe.contentWindow; }catch(_){ showFallback(); } }, 2500); }

function setMode(mode){
  currentMode = mode;
  // sections
  const secGlobal = document.getElementById('sec-global');
  const secBorough = document.getElementById('sec-borough');
  const secTime = document.getElementById('sec-time');
  // filters
  const boroughGroup = document.querySelector('#borough-chips')?.closest('.filter-group');
  const timeGroup = document.querySelector('#time-chips')?.closest('.filter-group');

  if(mode==='GLOBAL'){
    secGlobal.classList.remove('is-hidden');
    secBorough.classList.add('is-hidden');
    secTime.classList.add('is-hidden');
    boroughGroup?.classList.add('is-hidden');
    timeGroup?.classList.add('is-hidden');
  } else if(mode==='BOROUGH'){
    secGlobal.classList.add('is-hidden');
    secBorough.classList.remove('is-hidden');
    secTime.classList.add('is-hidden');
    boroughGroup?.classList.remove('is-hidden');
    timeGroup?.classList.add('is-hidden');
  } else {
    secGlobal.classList.add('is-hidden');
    secBorough.classList.add('is-hidden');
    secTime.classList.remove('is-hidden');
    boroughGroup?.classList.add('is-hidden');
    timeGroup?.classList.remove('is-hidden');
  }
}

async function main(){
  wireMapFallback();
  $('#reload-map').addEventListener('click', ()=>{ const f = document.getElementById('hotspot-map'); f.src = PATHS.map + `?t=${Date.now()}`; });
  let exec = await loadJSON(PATHS.executive); if(!exec) exec = await loadJSON(PATHS.execFallback);
  setLastUpdated(exec?.analysis_date); renderKpis(exec); renderPriorityPatterns(exec); renderRecommendations(exec);
  wirePatternModal(); await loadPatternGlobal();
}

window.addEventListener('DOMContentLoaded', main);
