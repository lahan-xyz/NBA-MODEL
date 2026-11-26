const fs = require('fs');
const games = JSON.parse(fs.readFileSync('./data/games.json','utf8'));
const factors = JSON.parse(fs.readFileSync('./data/factors.json','utf8'));

const F = new Map(factors.map(f => [f.game_id, f]));
const LG3 = {};
for (const s of ['2023-24','2024-25','2025-26','2026-27']) {
  let m = 0, a = 0;
  for (const f of factors) if (f.season === s) { m += f.h_FG3M + f.a_FG3M; a += f.h_FG3A + f.a_FG3A; }
  LG3[s] = a ? m/a : 0.36;
}

const ALPHA = 0.06, PACE_A = 0.15, INIT = 110.0, IPACE = 99.0;
const TRAIN = new Set(['2023-24']), VAL = new Set(['2024-25']), TEST = new Set(['2025-26']);

const adjPts = (f, side, lam, p3) => {
  const m3 = lam*f[side+'_FG3M'] + (1-lam)*f[side+'_FG3A']*p3;
  return 2*(f[side+'_FGM'] - f[side+'_FG3M']) + 3*m3 + f[side+'_FTM'];
};

function linfit(xs, ys) {
  const n = xs.length;
  let mx=0,my=0;
  for (let i=0;i<n;i++){mx+=xs[i];my+=ys[i];}
  mx/=n;my/=n;
  let num=0,den=0;
  for (let i=0;i<n;i++){num+=(xs[i]-mx)*(ys[i]-my);den+=(xs[i]-mx)**2;}
  const b = den===0?1:num/den;
  return {a:my-b*mx,b};
}

function b2bfit(rows, mFit) {
  const x=[],y=[];
  for (const r of rows){ x.push(r.g.away_b2b-r.g.home_b2b); y.push(r.am-(mFit.a+mFit.b*r.rawM)); }
  const n=x.length, mx=x.reduce((s,v)=>s+v,0)/n, my=y.reduce((s,v)=>s+v,0)/n;
  let num=0,den=0;
  for(let i=0;i<n;i++){num+=(x[i]-mx)*(y[i]-my);den+=(x[i]-mx)**2;}
  return den===0?0:num/den;
}

function run(lamO, lamD) {
  const R=new Map(); let league=INIT, missing=0;
  const get=t=>{let r=R.get(t); if(!r){r={o:INIT,d:INIT,p:IPACE,n:0};R.set(t,r);} return r;};
  const tr=[],va=[],te=[];

  for (const g of games) {
    const h=get(g.home), a=get(g.away);
    const pace=(h.p+a.p)/2;
    const hRaw=(h.o+a.d-league)*pace/100;
    const aRaw=(a.o+h.d-league)*pace/100;
    const rec={g,rawM:hRaw-aRaw,rawT:hRaw+aRaw,am:g.home_pts-g.away_pts,at:g.home_pts+g.away_pts};
    if(!g.ot){
      if(TRAIN.has(g.season)&&h.n>=10&&a.n>=10) tr.push(rec);
      else if(VAL.has(g.season)) va.push(rec);
      else if(TEST.has(g.season)) te.push(rec);
    }
    const f=F.get(g.game_id);
    let hO,aO,hD,aD;
    if (f) {
      const p3=LG3[g.season]||0.36;
      hO=adjPts(f,'h',lamO,p3)/g.poss*100;
      aO=adjPts(f,'a',lamO,p3)/g.poss*100;
      hD=adjPts(f,'a',lamD,p3)/g.poss*100;
      aD=adjPts(f,'h',lamD,p3)/g.poss*100;
    } else {
      missing++;
      hO=g.home_pts/g.poss*100; aO=g.away_pts/g.poss*100; hD=aO; aD=hO;
    }
    h.o=ALPHA*hO+(1-ALPHA)*h.o;
    a.o=ALPHA*aO+(1-ALPHA)*a.o;
    h.d=ALPHA*hD+(1-ALPHA)*h.d;
    a.d=ALPHA*aD+(1-ALPHA)*a.d;
    h.p=PACE_A*g.pace+(1-PACE_A)*h.p;
    a.p=PACE_A*g.pace+(1-PACE_A)*a.p;
    h.n++;a.n++;
    league=ALPHA*((hO+aO)/2)+(1-ALPHA)*league;
  }

  const fit = rows => {
    const mFit=linfit(rows.map(r=>r.rawM),rows.map(r=>r.am));
    const tFit=linfit(rows.map(r=>r.rawT),rows.map(r=>r.at));
    return {mFit,tFit,b1:b2bfit(rows,mFit)};
  };
  const score = (rows,C) => {
    let mAE=0,tAE=0,hits=0,bias=0;
    for(const r of rows){
      const m=C.mFit.a+C.mFit.b*r.rawM+C.b1*(r.g.away_b2b-r.g.home_b2b);
      const t=C.tFit.a+C.tFit.b*r.rawT;
      mAE+=Math.abs(m-r.am);tAE+=Math.abs(t-r.at);bias+=m-r.am;
      if((m>0)===(r.am>0))hits++;
    }
    const n=rows.length||1;
    return {mMAE:mAE/n,tMAE:tAE/n,acc:100*hits/n,bias:bias/n};
  };

  const vC=fit(tr), fC=fit(tr.concat(va));
  return {val:score(va,vC), test:score(te,fC), C:fC, missing};
}

const f2=(x,d=2)=>x.toFixed(d);
const cfgs=[
  ['baseline  O1.00 D1.00',1.00,1.00],
  ['defense   O1.00 D0.25',1.00,0.25],
  ['offense   O0.50 D1.00',0.50,1.00],
  ['both      O0.50 D0.25',0.50,0.25],
];
console.log('\nLAMBDA ABLATION — rest(B2B) on, SOS off, refit train+val');
console.log('config                    VAL mMAE | TEST mMAE  tMAE   acc%   bias  slope');
console.log('--------------------------------------------------------------------------');
const runs=cfgs.map(([n,o,d])=>({n,r:run(o,d)}));
for(const {n,r} of runs){
  console.log(`${n}    ${f2(r.val.mMAE)} |    ${f2(r.test.mMAE)}  ${f2(r.test.tMAE)}  ${f2(r.test.acc,1)}  ${f2(r.test.bias)}  ${f2(r.C.mFit.b)}`);
}
const best=runs.reduce((a,b)=>b.r.val.mMAE<a.r.val.mMAE?b:a);
console.log('--------------------------------------------------------------------------');
console.log(`Selected: ${best.n}   (unmatched factor rows: ${best.r.missing})`);
console.log(`  HCA ${f2(best.r.C.mFit.a)}   slope ${f2(best.r.C.mFit.b)}   B2B ${f2(best.r.C.b1)} pts`);
