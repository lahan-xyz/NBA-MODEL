const fs=require('fs');
const games=JSON.parse(fs.readFileSync('./data/games.json','utf8'));
const factors=JSON.parse(fs.readFileSync('./data/factors.json','utf8'));
const F=new Map(factors.map(f=>[f.game_id,f]));
const LAM_O=.5,LAM_D=.25,ALPHA=.06,PACE_A=.15,INIT=110,IPACE=99;
const FIT=new Set(['2023-24','2024-25']),TEST=new Set(['2025-26']);
const LG3={};
for(const s of new Set(factors.map(f=>f.season))){let m=0,a=0;
 for(const f of factors)if(f.season===s){m+=f.h_FG3M+f.a_FG3M;a+=f.h_FG3A+f.a_FG3A;}
 LG3[s]=a?m/a:.36;}
const adj=(f,sd,l,p)=>2*(f[sd+'_FGM']-f[sd+'_FG3M'])+3*(l*f[sd+'_FG3M']+(1-l)*f[sd+'_FG3A']*p)+f[sd+'_FTM'];
function linfit(x,y){const n=x.length;let mx=0,my=0;
 for(let i=0;i<n;i++){mx+=x[i];my+=y[i];}mx/=n;my/=n;let nu=0,de=0;
 for(let i=0;i<n;i++){nu+=(x[i]-mx)*(y[i]-my);de+=(x[i]-mx)**2;}
 const b=de===0?1:nu/de;return{a:my-b*mx,b};}
function logitFit(x,y){let a=0,b=0;
 for(let it=0;it<60;it++){let g0=0,g1=0,h00=0,h01=0,h11=0;
  for(let i=0;i<x.length;i++){const p=1/(1+Math.exp(-(a+b*x[i]))),d=p-y[i],w=p*(1-p);
   g0+=d;g1+=d*x[i];h00+=w;h01+=w*x[i];h11+=w*x[i]*x[i];}
  const det=h00*h11-h01*h01; if(Math.abs(det)<1e-12)break;
  const da=(h11*g0-h01*g1)/det,db=(h00*g1-h01*g0)/det;
  a-=da;b-=db; if(Math.abs(da)+Math.abs(db)<1e-9)break;}
 return{a,b};}
const Phi=z=>{const t=1/(1+.2316419*Math.abs(z)),d=.3989423*Math.exp(-z*z/2);
 const p=d*t*(.3193815+t*(-.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));
 return z>0?1-p:p;};

const R=new Map();let league=INIT;
const get=t=>{let r=R.get(t);if(!r){r={o:INIT,d:INIT,p:IPACE,n:0};R.set(t,r);}return r;};
const fr=[],te=[];
for(const g of games){
 const h=get(g.home),a=get(g.away),pace=(h.p+a.p)/2;
 const hR=(h.o+a.d-league)*pace/100,aR=(a.o+h.d-league)*pace/100;
 const rec={rawM:hR-aR,am:g.home_pts-g.away_pts,x:(g.away_b2b||0)-(g.home_b2b||0)};
 if(!g.ot){if(FIT.has(g.season)&&h.n>=10&&a.n>=10)fr.push(rec);else if(TEST.has(g.season))te.push(rec);}
 const f=F.get(g.game_id);let hO,aO,hD,aD;
 if(f){const p=LG3[g.season]||.36;
  hO=adj(f,'h',LAM_O,p)/g.poss*100;aO=adj(f,'a',LAM_O,p)/g.poss*100;
  hD=adj(f,'a',LAM_D,p)/g.poss*100;aD=adj(f,'h',LAM_D,p)/g.poss*100;
 }else{hO=g.home_pts/g.poss*100;aO=g.away_pts/g.poss*100;hD=aO;aD=hO;}
 h.o=ALPHA*hO+(1-ALPHA)*h.o;a.o=ALPHA*aO+(1-ALPHA)*a.o;
 h.d=ALPHA*hD+(1-ALPHA)*h.d;a.d=ALPHA*aD+(1-ALPHA)*a.d;
 h.p=PACE_A*g.pace+(1-PACE_A)*h.p;a.p=PACE_A*g.pace+(1-PACE_A)*a.p;
 h.n++;a.n++;league=ALPHA*((hO+aO)/2)+(1-ALPHA)*league;}
const M=linfit(fr.map(r=>r.rawM),fr.map(r=>r.am));
const b1=linfit(fr.map(r=>r.x),fr.map(r=>r.am-(M.a+M.b*r.rawM))).b;
const mg=r=>M.a+M.b*r.rawM+b1*r.x;
const sse=fr.reduce((s,r)=>s+(mg(r)-r.am)**2,0);
const sd=Math.sqrt(sse/fr.length);
const L=logitFit(fr.map(mg),fr.map(r=>r.am>0?1:0));
const base=fr.filter(r=>r.am>0).length/fr.length;

const models={
 'normal CDF (current)':r=>Phi(mg(r)/sd),
 'fitted logistic     ':r=>1/(1+Math.exp(-(L.a+L.b*mg(r)))),
 'constant base rate  ':()=>base};
console.log(`\nPROBABILITY CALIBRATION — test 2025-26, n=${te.length}`);
console.log(`residSD ${sd.toFixed(2)}   logistic a=${L.a.toFixed(3)} b=${L.b.toFixed(3)}`);
console.log('model                   Brier    logloss');
for(const [k,fn] of Object.entries(models)){
 let br=0,ll=0;
 for(const r of te){const p=Math.min(1-1e-9,Math.max(1e-9,fn(r))),y=r.am>0?1:0;
  br+=(p-y)**2; ll-=y*Math.log(p)+(1-y)*Math.log(1-p);}
 console.log(`${k}  ${(br/te.length).toFixed(5)}   ${(ll/te.length).toFixed(5)}`);}
console.log('\nlower is better. Adopt logistic ONLY if it beats normal CDF on both.');
