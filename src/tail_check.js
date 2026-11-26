const fs=require('fs');
const games=JSON.parse(fs.readFileSync('./data/games.json','utf8'));
const factors=JSON.parse(fs.readFileSync('./data/factors.json','utf8'));
const F=new Map(factors.map(f=>[f.game_id,f]));
const LAM_O=0.50,LAM_D=0.25,ALPHA=0.06,PACE_A=0.15,INIT=110,IPACE=99;
const FIT=new Set(['2023-24','2024-25']), TEST=new Set(['2025-26']);
const LG3={};
for(const s of new Set(factors.map(f=>f.season))){
  let m=0,a=0;
  for(const f of factors) if(f.season===s){m+=f.h_FG3M+f.a_FG3M;a+=f.h_FG3A+f.a_FG3A;}
  LG3[s]=a?m/a:0.36;
}
const adj=(f,side,lam,p3)=>{
  const m3=lam*f[side+'_FG3M']+(1-lam)*f[side+'_FG3A']*p3;
  return 2*(f[side+'_FGM']-f[side+'_FG3M'])+3*m3+f[side+'_FTM'];
};
function linfit(xs,ys){
  const n=xs.length;let mx=0,my=0;
  for(let i=0;i<n;i++){mx+=xs[i];my+=ys[i];}mx/=n;my/=n;
  let nu=0,de=0;
  for(let i=0;i<n;i++){nu+=(xs[i]-mx)*(ys[i]-my);de+=(xs[i]-mx)**2;}
  const b=de===0?1:nu/de;return{a:my-b*mx,b};
}
const R=new Map();let league=INIT;
const get=t=>{let r=R.get(t);if(!r){r={o:INIT,d:INIT,p:IPACE,n:0};R.set(t,r);}return r;};
const fitRows=[],test=[];
for(const g of games){
  const h=get(g.home),a=get(g.away);
  const pace=(h.p+a.p)/2;
  const hRaw=(h.o+a.d-league)*pace/100, aRaw=(a.o+h.d-league)*pace/100;
  const rec={g,rawM:hRaw-aRaw,am:g.home_pts-g.away_pts,x:(g.away_b2b||0)-(g.home_b2b||0)};
  if(!g.ot){
    if(FIT.has(g.season)&&h.n>=10&&a.n>=10) fitRows.push(rec);
    else if(TEST.has(g.season)) test.push(rec);
  }
  const f=F.get(g.game_id);
  let hO,aO,hD,aD;
  if(f){const p3=LG3[g.season]||0.36;
    hO=adj(f,'h',LAM_O,p3)/g.poss*100;aO=adj(f,'a',LAM_O,p3)/g.poss*100;
    hD=adj(f,'a',LAM_D,p3)/g.poss*100;aD=adj(f,'h',LAM_D,p3)/g.poss*100;
  } else {hO=g.home_pts/g.poss*100;aO=g.away_pts/g.poss*100;hD=aO;aD=hO;}
  h.o=ALPHA*hO+(1-ALPHA)*h.o;a.o=ALPHA*aO+(1-ALPHA)*a.o;
  h.d=ALPHA*hD+(1-ALPHA)*h.d;a.d=ALPHA*aD+(1-ALPHA)*a.d;
  h.p=PACE_A*g.pace+(1-PACE_A)*h.p;a.p=PACE_A*g.pace+(1-PACE_A)*a.p;
  h.n++;a.n++;league=ALPHA*((hO+aO)/2)+(1-ALPHA)*league;
}
const M=linfit(fitRows.map(r=>r.rawM),fitRows.map(r=>r.am));
const xs=fitRows.map(r=>r.x),ys=fitRows.map(r=>r.am-(M.a+M.b*r.rawM));
const mx=xs.reduce((s,v)=>s+v,0)/xs.length,my=ys.reduce((s,v)=>s+v,0)/ys.length;
let nu=0,de=0;
for(let i=0;i<xs.length;i++){nu+=(xs[i]-mx)*(ys[i]-my);de+=(xs[i]-mx)**2;}
const b1=de===0?0:nu/de;

const edges=[0,3,6,9,12,15,99];
const bins=edges.slice(0,-1).map((lo,i)=>({lo,hi:edges[i+1],p:[],a:[],w:0}));
for(const r of test){
  const pred=M.a+M.b*r.rawM+b1*r.x;
  const s=pred>=0?1:-1;
  const fav=s*pred, act=s*r.am;
  const b=bins.find(b=>fav>=b.lo&&fav<b.hi);
  if(!b) continue;
  b.p.push(fav); b.a.push(act); if(act>0) b.w++;
}
console.log('\nTAIL CALIBRATION — oriented to predicted favourite, 2025-26');
console.log('bin        n   pred   actual    gap    win%   SE');
for(const b of bins){
  const n=b.p.length; if(!n){console.log(`${b.lo}-${b.hi}`.padEnd(9)+'   0');continue;}
  const mp=b.p.reduce((s,v)=>s+v,0)/n, ma=b.a.reduce((s,v)=>s+v,0)/n;
  const sd=Math.sqrt(b.a.reduce((s,v)=>s+(v-ma)**2,0)/n);
  console.log(`${(b.lo+'-'+b.hi).padEnd(9)}${String(n).padStart(4)}  ${mp.toFixed(2).padStart(6)} ${ma.toFixed(2).padStart(7)} ${(ma-mp).toFixed(2).padStart(7)} ${(100*b.w/n).toFixed(1).padStart(6)} ${(sd/Math.sqrt(n)).toFixed(2).padStart(6)}`);
}
console.log('\ngap < 0 means the model over-predicts favourites in that bin.');
console.log('Only trust a gap bigger than ~2x its SE.');
