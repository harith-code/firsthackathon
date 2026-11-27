/* ========== helpers ========== */
const rnd=(s=>()=>{s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296})(42);
const gauss=()=>Math.sqrt(-2*Math.log(rnd()+1e-9))*Math.cos(6.283*rnd());
const clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x));
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(x=>(x-m)**2)))||1};
const pct=x=>(x*100).toFixed(0)+'%';
const $=id=>document.getElementById(id);
const SUBJ=['Maths','Programming','DBMS','Networks','Data Science'];
const SUBOFF=[-.06,.03,0,-.03,.04];
const N=120, WEEK_NOW=9, WEEKS=15, CUT=.75;

/* ========== 1. RAW DATA (synthetic; replace with your CSV) ========== */
// Each record = one student in one semester: 15 weekly attendance rates + internal marks.
const raw=[];
for(let i=1;i<=N;i++){
  let base=clamp(.82+.12*gauss(),.35,.99);
  const group='ABC'[i%3], drift=.012*gauss();
  for(let s=1;s<=3;s++){
    base=clamp(base+drift+.02*gauss(),.3,.99);
    const marks=Math.round(clamp(35+60*(base-.4)+8*gauss(),10,98));
    const weekly=[...Array(WEEKS)].map((_,w)=>clamp(base+drift*(w-7)+.1*gauss()));
    if(rnd()<.04) weekly[Math.floor(rnd()*WEEKS)]=null;   // missing value
    if(rnd()<.02) weekly[Math.floor(rnd()*WEEKS)]=1.2;    // bad value
    raw.push({id:i,group,sem:s,weekly,marks});
  }
}

/* ========== 2. PREPROCESSING ========== */
let nulls=0,fixed=0;
const data=raw.map(r=>{
  const m=mean(r.weekly.filter(x=>x!==null).map(x=>Math.min(x,1)));
  return{...r,weekly:r.weekly.map(x=>{
    if(x===null){nulls++;return m}
    if(x>1||x<0){fixed++;return clamp(x)}
    return x})};
});
// feature engineering: only what is known at week 9
const KEYS=['att','recent','slope','marks','prev','low'];
const LABEL={att:'Attendance so far',recent:'Last 3 weeks',slope:'Trend (slope)',marks:'Internal marks',prev:'Previous sem attendance',low:'Weeks below 60%'};
data.forEach(r=>{
  const w=r.weekly, early=w.slice(0,WEEK_NOW), prev=data.find(d=>d.id===r.id&&d.sem===r.sem-1);
  r.f={att:mean(early),recent:mean(w.slice(6,9)),slope:mean(w.slice(6,9))-mean(w.slice(0,3)),
       marks:r.marks,prev:prev?mean(prev.weekly):mean(early),low:early.filter(x=>x<.6).length};
  r.final=mean(w); r.y=r.final<CUT?1:0;
});
const train=data.filter(r=>r.sem<3), test=data.filter(r=>r.sem===3);

/* ========== 3. ATTENDANCE ANALYTICS ========== */
const subjectAtt=SUBJ.map((_,k)=>mean(test.map(r=>clamp(r.f.att+SUBOFF[k]+.03*gauss()))));
const semAtt=[1,2,3].map(s=>mean(data.filter(r=>r.sem===s).map(r=>r.final)));
const groupWeekly=['A','B','C'].map(g=>{
  const rs=test.filter(r=>r.group===g);
  return[...Array(WEEK_NOW)].map((_,w)=>mean(rs.map(r=>r.weekly[w])));
});
function pearson(x,y){const mx=mean(x),my=mean(y);
  return x.reduce((s,v,i)=>s+(v-mx)*(y[i]-my),0)/Math.sqrt(x.reduce((s,v)=>s+(v-mx)**2,0)*y.reduce((s,v)=>s+(v-my)**2,0))}
const corr=pearson(test.map(r=>r.f.att),test.map(r=>r.marks));

/* ========== 4. ML: logistic regression (gradient descent) ========== */
const sig=z=>1/(1+Math.exp(-z));
const mu=KEYS.map(k=>mean(train.map(r=>r.f[k]))), sg=KEYS.map(k=>sd(train.map(r=>r.f[k])));
const vec=r=>KEYS.map((k,j)=>(r.f[k]-mu[j])/sg[j]);
const X=train.map(vec), Y=train.map(r=>r.y);
let W=Array(KEYS.length).fill(0), B=0;
for(let it=0;it<1000;it++){
  const gw=W.map(()=>0);let gb=0;
  X.forEach((x,i)=>{const e=sig(B+x.reduce((s,v,j)=>s+v*W[j],0))-Y[i];x.forEach((v,j)=>gw[j]+=e*v);gb+=e});
  W=W.map((w,j)=>w-.1*gw[j]/X.length);B-=.1*gb/X.length;
}
const prob=r=>sig(B+vec(r).reduce((s,v,j)=>s+v*W[j],0));
test.forEach(r=>r.p=prob(r));

// evaluation
let TP=0,FP=0,FN=0,TN=0;
test.forEach(r=>{const p=r.p>=.5?1:0;if(p&&r.y)TP++;else if(p&&!r.y)FP++;else if(!p&&r.y)FN++;else TN++});
const acc=(TP+TN)/test.length, prec=TP/(TP+FP||1), rec=TP/(TP+FN||1), f1=2*prec*rec/(prec+rec||1);

/* ========== 5. ALERT + RECOMMENDATION ========== */
const level=p=>p>=.7?'High':p>=.4?'Medium':'Low';
function recommend(r){
  const need=(CUT*WEEKS-r.f.att*WEEK_NOW)/(WEEKS-WEEK_NOW), out=[];
  if(need>1) out.push('Cannot reach 75% by attending alone: start condonation or medical-leave review');
  else if(need>0) out.push(`Attend at least ${Math.ceil(need*100)}% of the remaining sessions`);
  else out.push('On track: keep attending regularly');
  if(r.f.slope<-.05) out.push('Attendance is falling: meet mentor this week');
  if(r.f.att<.65) out.push('Counselling session and parent notification');
  if(r.f.marks<50) out.push('Enrol in remedial classes');
  if(r.f.low>=3) out.push('Check for recurring causes (travel, health, timetable clash)');
  return out;
}
test.forEach(r=>{r.level=level(r.p);r.rec=recommend(r);r.def=r.f.att<CUT});

/* ========== 6. DASHBOARD ========== */
const high=test.filter(r=>r.level==='High');
$('pipeline').innerHTML=[
  ['Raw data',`${raw.length} records`],['Preprocessing',`${nulls} filled, ${fixed} clipped`],
  ['Attendance analytics','subject, semester, group'],['ML prediction',`accuracy ${pct(acc)}`],
  ['Alert',`${high.length} high-risk students`],['Recommendation','per student']
].map(([a,b])=>`<div class="step"><b>${a}</b><small>${b}</small></div>`).join('');

$('kpis').innerHTML=[
  ['Students',N,''],['Average attendance now',pct(mean(test.map(r=>r.f.att))),''],
  ['Defaulters (below 75%)',test.filter(r=>r.def).length,'bad'],
  ['Predicted shortage',test.filter(r=>r.p>=.5).length,'bad'],['Model F1 score',f1.toFixed(2),'']
].map(([a,b,c])=>`<div class="kpi ${c}"><span>${a}</span><strong>${b}</strong></div>`).join('');

const css=getComputedStyle(document.documentElement);
const col=['#0f6b57','#d98e04','#c0392b','#3b6fb6','#7a4fa3'];
const opt=(extra={})=>({responsive:true,plugins:{legend:{display:extra.legend||false}},...extra.o});
const pctAxis={ticks:{callback:v=>v+'%'}};
new Chart($('cSubject'),{type:'bar',data:{labels:SUBJ,datasets:[{data:subjectAtt.map(v=>+(v*100).toFixed(1)),backgroundColor:SUBJ.map((_,i)=>subjectAtt[i]<CUT?'#c0392b':'#0f6b57')}]},options:opt({o:{scales:{y:{min:50,max:100,...pctAxis}}}})});
new Chart($('cSem'),{type:'line',data:{labels:['Sem 1','Sem 2','Sem 3'],datasets:[{data:semAtt.map(v=>+(v*100).toFixed(1)),borderColor:'#0f6b57',tension:.3}]},options:opt({o:{scales:{y:{min:50,max:100,...pctAxis}}}})});
new Chart($('cWeek'),{type:'line',data:{labels:[...Array(WEEK_NOW)].map((_,i)=>'W'+(i+1)),datasets:groupWeekly.map((g,i)=>({label:'Group '+'ABC'[i],data:g.map(v=>+(v*100).toFixed(1)),borderColor:col[i],tension:.3}))},options:opt({legend:true,o:{scales:{y:{min:50,max:100,...pctAxis}}}})});
$('corr').textContent=`correlation r = ${corr.toFixed(2)}`;
new Chart($('cScatter'),{type:'scatter',data:{datasets:[{data:test.map(r=>({x:+(r.f.att*100).toFixed(1),y:r.marks})),backgroundColor:test.map(r=>r.def?'#c0392b':'#0f6b57')}]},options:opt({o:{scales:{x:{title:{display:true,text:'Attendance %'}},y:{title:{display:true,text:'Marks'}}}}})});
new Chart($('cWeights'),{type:'bar',data:{labels:KEYS.map(k=>LABEL[k]),datasets:[{data:W.map(w=>+w.toFixed(2)),backgroundColor:W.map(w=>w>0?'#c0392b':'#0f6b57')}]},options:opt({o:{indexAxis:'y'}})});

$('metrics').innerHTML=[['Accuracy',pct(acc)],['Precision',pct(prec)],['Recall',pct(rec)],['F1 score',f1.toFixed(2)]]
  .map(([a,b])=>`<div><strong>${b}</strong>${a}</div>`).join('');
$('cm').innerHTML=`<tr><th></th><th>Predicted OK</th><th>Predicted shortage</th></tr>
<tr><th>Actually OK</th><td>${TN}</td><td>${FP}</td></tr><tr><th>Actually short</th><td>${FN}</td><td>${TP}</td></tr>`;

$('alerts').innerHTML=[...high].sort((a,b)=>b.p-a.p).slice(0,6).map(r=>
  `<li><b>S${String(r.id).padStart(3,'0')}</b> (Group ${r.group}): ${pct(r.f.att)} attendance, ${pct(r.p)} chance of finishing below 75%. ${r.rec[0]}.</li>`).join('')||'<li>No high-risk students.</li>';

function render(){
  const q=$('search').value.trim().toUpperCase(), f=$('filter').value;
  const rows=[...test].sort((a,b)=>b.p-a.p).filter(r=>
    (!q||('S'+String(r.id).padStart(3,'0')).includes(q))&&
    (f==='all'||(f==='def'?r.def:r.level===f)));
  $('tbl').tBodies[0].innerHTML=rows.map(r=>`<tr>
    <td>S${String(r.id).padStart(3,'0')}</td><td>${r.group}</td><td>${pct(r.f.att)}</td><td>${pct(r.f.recent)}</td>
    <td>${r.marks}</td><td>${pct(r.p)}</td><td><span class="badge ${r.level}">${r.level}</span></td>
    <td>${r.rec.join('. ')}.</td></tr>`).join('')||'<tr><td colspan="8">No students match this filter.</td></tr>';
}
$('search').oninput=render;$('filter').onchange=render;render();

$('csv').onclick=()=>{
  const rows=[['Student','Group','Attendance','Risk','Level','Recommendation'],
    ...test.filter(r=>r.def||r.level!=='Low').map(r=>['S'+r.id,r.group,pct(r.f.att),pct(r.p),r.level,r.rec.join('. ')])];
  const blob=new Blob([rows.map(r=>r.map(c=>`"${c}"`).join(',')).join('\n')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='defaulters.csv';a.click();
};
