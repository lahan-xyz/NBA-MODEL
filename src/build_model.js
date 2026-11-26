const fs = require('fs');
const games = JSON.parse(fs.readFileSync('./data/games.json','utf8'));
const factors = JSON.parse(fs.readFileSync('./data/factors.json','utf8'));
const F = new Map(factors.map(f => [f.game_id, f]));

const LAM_O = 0.50, LAM_D = 0.25, ALPHA = 0.06, PACE_A = 0.15;
const INIT = 110.0, IPACE = 99.0;
const DONE = new Set(['2023-24','2024-25','2025-26']);

const LG3 = {};
for (const s of new Set(factors.map(f=>f.season))) {
  let m=0,a=0;
  for (const f of factors) if (f.season===s){m+=f.h_FG3M+f.a_FG3M;a+=f.h_FG3A+f.a_FG3A;}
  LG3[s]=a?m/a:0.36;
}
const adjPts=(f,side,lam,p3)=>{
  const m3=lam*f[side+'_FG3M']+(1-lam)*f[side+'_FG3A']*p3;
  return 2*(f[side+'_FGM']-f[side+'_FG3M'])+3*m3+f[side+'_FTM'];
};
function linfit(xs,ys){
  const n=xs.length; let mx=0,my=0;
  for(let i=0;i<n;i++){mx+=xs[i];my+=ys[i];} mx/=n;my/=n;
  let num=0,den=0;
  for(let i=0;i<n;i++){num+=(xs[i]-mx)*(ys[i]-my);den+=(xs[i]-mx)**2;}
  const b=den===0?1:num/den; return {a:my-b*mx,b};
}

const R=new Map(); let league=INIT;
const get=t=>{let r=R.get(t); if(!r){r={o:INIT,d:INIT,p:IPACE,n:0};R.set(t,r);} return r;};
const rows=[];

for (const g of games) {
  const h=get(g.home), a=get(g.away);
  const pace=(h.p+a.p)/2;
  const hRaw=(h.o+a.d-league)*pace/100;
  const aRaw=(a.o+h.d-league)*pace/100;
  if(!g.ot && DONE.has(g.season) && h.n>=10 && a.n>=10)
    rows.push({rawM:hRaw-aRaw,rawT:hRaw+aRaw,am:g.home_pts-g.away_pts,
               at:g.home_pts+g.away_pts,x:(g.away_b2b||0)-(g.home_b2b||0)});
  const f=F.get(g.game_id);
  let hO,aO,hD,aD;
  if(f){
    const p3=LG3[g.season]||0.36;
    hO=adjPts(f,'h',LAM_O,p3)/g.poss*100; aO=adjPts(f,'a',LAM_O,p3)/g.poss*100;
    hD=adjPts(f,'a',LAM_D,p3)/g.poss*100; aD=adjPts(f,'h',LAM_D,p3)/g.poss*100;
  } else { hO=g.home_pts/g.poss*100; aO=g.away_pts/g.poss*100; hD=aO; aD=hO; }
  h.o=ALPHA*hO+(1-ALPHA)*h.o; a.o=ALPHA*aO+(1-ALPHA)*a.o;
  h.d=ALPHA*hD+(1-ALPHA)*h.d; a.d=ALPHA*aD+(1-ALPHA)*a.d;
  h.p=PACE_A*g.pace+(1-PACE_A)*h.p; a.p=PACE_A*g.pace+(1-PACE_A)*a.p;
  h.n++;a.n++;
  league=ALPHA*((hO+aO)/2)+(1-ALPHA)*league;
}

const mFit=linfit(rows.map(r=>r.rawM),rows.map(r=>r.am));
const tFit=linfit(rows.map(r=>r.rawT),rows.map(r=>r.at));
const xs=rows.map(r=>r.x), ys=rows.map(r=>r.am-(mFit.a+mFit.b*r.rawM));
const mx=xs.reduce((s,v)=>s+v,0)/xs.length, my=ys.reduce((s,v)=>s+v,0)/ys.length;
let num=0,den=0;
for(let i=0;i<xs.length;i++){num+=(xs[i]-mx)*(ys[i]-my);den+=(xs[i]-mx)**2;}
const b1=den===0?0:num/den;

let sse=0;
for(const r of rows){const m=mFit.a+mFit.b*r.rawM+b1*r.x; sse+=(m-r.am)**2;}
const residSD=Math.sqrt(sse/rows.length);

fs.writeFileSync('./data/model.json',JSON.stringify({
  lamO:LAM_O, lamD:LAM_D, alpha:ALPHA, paceAlpha:PACE_A,
  leagueAvg:league, marginFit:mFit, totalFit:tFit, b2b:b1, residSD,
  calibratedOn:rows.length, asOf:games[games.length-1].date,
  ratings:Object.fromEntries(R)
},null,2));

console.log(`Calibrated on ${rows.length} completed-season games`);
console.log(`HCA ${mFit.a.toFixed(2)}  slope ${mFit.b.toFixed(2)}  B2B ${b1.toFixed(2)}  residSD ${residSD.toFixed(2)}`);
console.log(`Ratings as of ${games[games.length-1].date} -> data/model.json`);
