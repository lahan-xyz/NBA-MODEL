const fs = require('fs');
const games = JSON.parse(fs.readFileSync('./data/games.json', 'utf8'));
if (games[0].home_b2b === undefined) {
  console.error('Run: python add_features.py first'); process.exit(1);
}

const ALPHA = 0.06, PACE_A = 0.15, INIT = 110.0, IPACE = 99.0;
const TRAIN = new Set(['2023-24']), VAL = new Set(['2024-25']), TEST = new Set(['2025-26']);

function linfit(xs, ys) {
  const n = xs.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xs[i]-mx)*(ys[i]-my); den += (xs[i]-mx)**2; }
  const b = den === 0 ? 1 : num/den;
  return { a: my - b*mx, b };
}

function restfit(rows, mFit) {
  const x1 = [], x2 = [], y = [];
  for (const r of rows) {
    x1.push(r.g.away_b2b - r.g.home_b2b);
    x2.push(r.g.away_3in4 - r.g.home_3in4);
    y.push(r.am - (mFit.a + mFit.b * r.rawM));
  }
  const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
  const m1 = mean(x1), m2 = mean(x2), my = mean(y);
  let S11=0,S22=0,S12=0,S1y=0,S2y=0;
  for (let i=0;i<y.length;i++){
    const a=x1[i]-m1, b=x2[i]-m2, c=y[i]-my;
    S11+=a*a; S22+=b*b; S12+=a*b; S1y+=a*c; S2y+=b*c;
  }
  const det = S11*S22 - S12*S12;
  if (Math.abs(det) < 1e-9) return { b1:0, b2:0 };
  return { b1:(S22*S1y - S12*S2y)/det, b2:(S11*S2y - S12*S1y)/det };
}

function run(cfg) {
  const R = new Map();
  let league = INIT;
  const get = t => {
    let r = R.get(t);
    if (!r) { r = {o:INIT,d:INIT,p:IPACE,n:0}; R.set(t,r); }
    return r;
  };
  const tr=[], va=[], te=[];

  for (const g of games) {
    const h = get(g.home), a = get(g.away);
    const pace = (h.p + a.p)/2;
    const hRaw = (h.o + a.d - league)*pace/100;
    const aRaw = (a.o + h.d - league)*pace/100;
    const rec = { g, rawM:hRaw-aRaw, rawT:hRaw+aRaw,
                  am:g.home_pts-g.away_pts, at:g.home_pts+g.away_pts };
    if (!g.ot) {
      if (TRAIN.has(g.season) && h.n>=10 && a.n>=10) tr.push(rec);
      else if (VAL.has(g.season)) va.push(rec);
      else if (TEST.has(g.season)) te.push(rec);
    }
    const ho0=h.o, ao0=a.o, hd0=h.d, ad0=a.d;
    const hO = g.home_pts/g.poss*100, aO = g.away_pts/g.poss*100;
    const hOa = cfg.sos ? hO + league - ad0 : hO;
    const aOa = cfg.sos ? aO + league - hd0 : aO;
    const hDa = cfg.sos ? aO + league - ao0 : aO;
    const aDa = cfg.sos ? hO + league - ho0 : hO;
    h.o = ALPHA*hOa + (1-ALPHA)*h.o;
    a.o = ALPHA*aOa + (1-ALPHA)*a.o;
    h.d = ALPHA*hDa + (1-ALPHA)*h.d;
    a.d = ALPHA*aDa + (1-ALPHA)*a.d;
    h.p = PACE_A*g.pace + (1-PACE_A)*h.p;
    a.p = PACE_A*g.pace + (1-PACE_A)*a.p;
    h.n++; a.n++;
    league = ALPHA*((hO+aO)/2) + (1-ALPHA)*league;
  }

  const fit = rows => {
    const mFit = linfit(rows.map(r=>r.rawM), rows.map(r=>r.am));
    const tFit = linfit(rows.map(r=>r.rawT), rows.map(r=>r.at));
    return { mFit, tFit, rest: cfg.rest ? restfit(rows, mFit) : {b1:0,b2:0} };
  };
  const score = (rows, F) => {
    let mAE=0,tAE=0,hits=0,bias=0;
    for (const r of rows) {
      const m = F.mFit.a + F.mFit.b*r.rawM
              + F.rest.b1*(r.g.away_b2b - r.g.home_b2b)
              + F.rest.b2*(r.g.away_3in4 - r.g.home_3in4);
      const t = F.tFit.a + F.tFit.b*r.rawT;
      mAE+=Math.abs(m-r.am); tAE+=Math.abs(t-r.at); bias+=m-r.am;
      if ((m>0)===(r.am>0)) hits++;
    }
    const n = rows.length||1;
    return { mMAE:mAE/n, tMAE:tAE/n, acc:100*hits/n, bias:bias/n };
  };

  const vFit = fit(tr);
  const fFit = fit(tr.concat(va));   // refit on train+val before test
  return { cfg, val:score(va,vFit), test:score(te,fFit), F:fFit };
}

const f=(x,d=2)=>x.toFixed(d);
const configs = [
  {name:'baseline      ', sos:false, rest:false},
  {name:'+ schedule    ', sos:true,  rest:false},
  {name:'+ rest        ', sos:false, rest:true },
  {name:'+ both        ', sos:true,  rest:true },
];

console.log('\nABLATION — select on VAL, refit train+val, report TEST');
console.log('config           VAL mMAE | TEST mMAE  tMAE   acc%   bias');
console.log('----------------------------------------------------------');
const runs = configs.map(c => ({...c, r:run(c)}));
for (const c of runs) {
  console.log(`${c.name}    ${f(c.r.val.mMAE)} |    ${f(c.r.test.mMAE)}  ${f(c.r.test.tMAE)}  ${f(c.r.test.acc,1)}  ${f(c.r.test.bias)}`);
}
const best = runs.reduce((a,b)=> b.r.val.mMAE < a.r.val.mMAE ? b : a);
console.log('----------------------------------------------------------');
console.log(`Selected: ${best.name.trim()}`);
console.log(`  HCA intercept : ${f(best.r.F.mFit.a)}`);
console.log(`  margin slope  : ${f(best.r.F.mFit.b)}`);
console.log(`  B2B penalty   : ${f(best.r.F.rest.b1)} pts`);
console.log(`  3-in-4 penalty: ${f(best.r.F.rest.b2)} pts`);
