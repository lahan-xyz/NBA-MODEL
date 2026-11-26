const fs = require('fs');
const games = JSON.parse(fs.readFileSync('./data/games.json', 'utf8'));

const INIT_RTG = 110.0, INIT_PACE = 99.0;
const TRAIN = new Set(['2023-24']);
const VAL   = new Set(['2024-25']);
const TEST  = new Set(['2025-26']);

function linfit(xs, ys) {
  const n = xs.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) * (xs[i] - mx);
  }
  const b = den === 0 ? 1 : num / den;
  return { a: my - b * mx, b };
}

function evaluate(alpha, paceAlpha) {
  const R = new Map();
  let league = INIT_RTG;
  const get = (t) => {
    let r = R.get(t);
    if (!r) { r = { o: INIT_RTG, d: INIT_RTG, p: INIT_PACE, n: 0 }; R.set(t, r); }
    return r;
  };

  const txm = [], tym = [], txt = [], tyt = [];
  const val = [], test = [];

  for (const g of games) {
    const h = get(g.home), a = get(g.away);
    const pace = (h.p + a.p) / 2;
    const hRaw = (h.o + a.d - league) * pace / 100;
    const aRaw = (a.o + h.d - league) * pace / 100;
    const rawM = hRaw - aRaw, rawT = hRaw + aRaw;
    const am = g.home_pts - g.away_pts, at = g.home_pts + g.away_pts;

    if (!g.ot) {
      if (TRAIN.has(g.season) && h.n >= 10 && a.n >= 10) {
        txm.push(rawM); tym.push(am); txt.push(rawT); tyt.push(at);
      } else if (VAL.has(g.season)) {
        val.push({ rawM, rawT, am, at });
      } else if (TEST.has(g.season)) {
        test.push({ rawM, rawT, am, at, g });
      }
    }

    const hO = g.home_pts / g.poss * 100, aO = g.away_pts / g.poss * 100;
    h.o = alpha * hO + (1 - alpha) * h.o;
    h.d = alpha * aO + (1 - alpha) * h.d;
    a.o = alpha * aO + (1 - alpha) * a.o;
    a.d = alpha * hO + (1 - alpha) * a.d;
    h.p = paceAlpha * g.pace + (1 - paceAlpha) * h.p;
    a.p = paceAlpha * g.pace + (1 - paceAlpha) * a.p;
    h.n++; a.n++;
    league = alpha * ((hO + aO) / 2) + (1 - alpha) * league;
  }

  const mFit = linfit(txm, tym), tFit = linfit(txt, tyt);
  const score = (rows) => {
    let mAE = 0, tAE = 0, hits = 0, bias = 0;
    for (const t of rows) {
      const m = mFit.a + mFit.b * t.rawM;
      const tt = tFit.a + tFit.b * t.rawT;
      mAE += Math.abs(m - t.am);
      tAE += Math.abs(tt - t.at);
      bias += m - t.am;
      if ((m > 0) === (t.am > 0)) hits++;
    }
    const n = rows.length || 1;
    return { mMAE: mAE / n, tMAE: tAE / n, acc: 100 * hits / n, bias: bias / n };
  };

  return { alpha, paceAlpha, mFit, tFit, league, ratings: R,
           mSlope: mFit.b, tSlope: tFit.b,
           val: score(val), test: score(test), nTest: test.length };
}

const f = (x, d = 2) => x.toFixed(d);
const pad = (s, w) => String(s).padStart(w);

console.log('\nALPHA SWEEP — select on 2024-25, report on 2025-26');
console.log('alpha   mSlope |  VAL mMAE | TEST mMAE  tMAE   acc%');
console.log('------------------------------------------------------');
const alphas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30];
const runs = alphas.map(a => evaluate(a, a));
for (const r of runs) {
  console.log(
    `${pad(f(r.alpha), 5)}   ${pad(f(r.mSlope), 6)} |` +
    `     ${pad(f(r.val.mMAE), 5)} |` +
    `     ${pad(f(r.test.mMAE), 5)}  ${pad(f(r.test.tMAE), 5)}  ${pad(f(r.test.acc, 1), 5)}`
  );
}

let best = runs.reduce((a, b) => (b.val.mMAE < a.val.mMAE ? b : a));
console.log('------------------------------------------------------');
console.log(`Selected on validation: alpha = ${best.alpha}`);

console.log('\nPACE ALPHA SWEEP at alpha = ' + best.alpha);
console.log('paceA  |  VAL tMAE | TEST tMAE  mMAE');
const paceRuns = [0.02, 0.04, 0.08, 0.15, 0.30].map(p => evaluate(best.alpha, p));
for (const r of paceRuns) {
  console.log(`${pad(f(r.paceAlpha), 5)}  |     ${pad(f(r.val.tMAE), 5)} |` +
              `     ${pad(f(r.test.tMAE), 5)}  ${pad(f(r.test.mMAE), 5)}`);
}
const bestPace = paceRuns.reduce((a, b) => (b.val.tMAE < a.val.tMAE ? b : a));

console.log('\n===============================================');
console.log(`FINAL: alpha=${bestPace.alpha}  paceAlpha=${bestPace.paceAlpha}`);
console.log('===============================================');
console.log(`Margin fit      : ${f(bestPace.mFit.a)} + ${f(bestPace.mFit.b)} * raw`);
console.log(`Total fit       : ${f(bestPace.tFit.a)} + ${f(bestPace.tFit.b)} * raw`);
console.log(`TEST margin MAE : ${f(bestPace.test.mMAE)}`);
console.log(`TEST total MAE  : ${f(bestPace.test.tMAE)}`);
console.log(`TEST winner acc : ${f(bestPace.test.acc, 1)}%`);
console.log(`TEST bias       : ${f(bestPace.test.bias)}`);
console.log('===============================================');

fs.writeFileSync('./data/model.json', JSON.stringify({
  alpha: bestPace.alpha,
  paceAlpha: bestPace.paceAlpha,
  leagueAvg: bestPace.league,
  marginFit: bestPace.mFit,
  totalFit: bestPace.tFit,
  asOf: games[games.length - 1].date,
  ratings: Object.fromEntries(bestPace.ratings)
}, null, 2));
console.log('\nBest config written to data/model.json');
