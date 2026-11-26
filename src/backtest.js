const fs = require('fs');
const games = JSON.parse(fs.readFileSync('./data/games.json', 'utf8'));

const ALPHA = 0.08;                 // EWMA decay, ~8 game half-life
const INIT_RTG = 110.0;
const INIT_PACE = 99.0;
const TRAIN = new Set(['2023-24', '2024-25']);
const TEST  = new Set(['2025-26']);

const ratings = new Map();
let league = INIT_RTG;

function get(team) {
  if (!ratings.has(team)) {
    ratings.set(team, { ortg: INIT_RTG, drtg: INIT_RTG, pace: INIT_PACE, n: 0 });
  }
  return ratings.get(team);
}

// League-relative matchup model (no lookahead)
function predict(homeName, awayName) {
  const h = get(homeName), a = get(awayName);
  const pace = (h.pace + a.pace) / 2;
  const hRaw = (h.ortg + a.drtg - league) * pace / 100;
  const aRaw = (a.ortg + h.drtg - league) * pace / 100;
  return { total: hRaw + aRaw, margin: hRaw - aRaw, pace };
}

function update(g) {
  const h = get(g.home), a = get(g.away);
  const hO = g.home_pts / g.poss * 100;
  const aO = g.away_pts / g.poss * 100;
  h.ortg = ALPHA * hO + (1 - ALPHA) * h.ortg;
  h.drtg = ALPHA * aO + (1 - ALPHA) * h.drtg;
  a.ortg = ALPHA * aO + (1 - ALPHA) * a.ortg;
  a.drtg = ALPHA * hO + (1 - ALPHA) * a.drtg;
  h.pace = ALPHA * g.pace + (1 - ALPHA) * h.pace;
  a.pace = ALPHA * g.pace + (1 - ALPHA) * a.pace;
  h.n++; a.n++;
  league = ALPHA * ((hO + aO) / 2) + (1 - ALPHA) * league;
}

function linfit(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const b = den === 0 ? 1 : num / den;
  return { a: my - b * mx, b };
}

const txm = [], tym = [], txt = [], tyt = [];
const test = [];

for (const g of games) {
  const p = predict(g.home, g.away);
  const am = g.home_pts - g.away_pts;
  const at = g.home_pts + g.away_pts;
  if (!g.ot) {
    if (TRAIN.has(g.season)) {
      txm.push(p.margin); tym.push(am);
      txt.push(p.total);  tyt.push(at);
    } else if (TEST.has(g.season)) {
      test.push({ g, p, am, at });
    }
  }
  update(g);   // always update, after predicting
}

if (!txm.length || !test.length) {
  console.error('Not enough data. Check that fetch_games.py pulled multiple seasons.');
  process.exit(1);
}

const mFit = linfit(txm, tym);
const tFit = linfit(txt, tyt);

let mAE = 0, tAE = 0, pAE = 0, hits = 0, bias = 0;
let baseMAE = 0, baseHits = 0;
const baseTotal = tyt.reduce((s, v) => s + v, 0) / tyt.length;

for (const t of test) {
  const margin = mFit.a + mFit.b * t.p.margin;
  const total  = tFit.a + tFit.b * t.p.total;
  const hp = (total + margin) / 2;
  const ap = (total - margin) / 2;

  mAE += Math.abs(margin - t.am);
  tAE += Math.abs(total - t.at);
  pAE += (Math.abs(hp - t.g.home_pts) + Math.abs(ap - t.g.away_pts)) / 2;
  bias += margin - t.am;
  if ((margin > 0) === (t.am > 0)) hits++;

  baseMAE += Math.abs(2.5 - t.am);
  if (t.am > 0) baseHits++;
}

const n = test.length;
const f = (x) => x.toFixed(2);

console.log('\n===============================================');
console.log('  BACKTEST — held out season 2025-26');
console.log('===============================================');
console.log(`Games evaluated : ${n}   (OT excluded)`);
console.log(`Trained on      : ${txm.length} games`);
console.log('-----------------------------------------------');
console.log(`Margin fit      : ${f(mFit.a)} + ${f(mFit.b)} * raw`);
console.log(`Total fit       : ${f(tFit.a)} + ${f(tFit.b)} * raw`);
console.log('-----------------------------------------------');
console.log(`Margin MAE      : ${f(mAE / n)} pts`);
console.log(`Total  MAE      : ${f(tAE / n)} pts`);
console.log(`Team pts MAE    : ${f(pAE / n)} pts`);
console.log(`Winner accuracy : ${f(100 * hits / n)}%`);
console.log(`Margin bias     : ${f(bias / n)} pts`);
console.log('-----------------------------------------------');
console.log('BASELINES (must beat these):');
console.log(`Always home -2.5: ${f(baseMAE / n)} MAE, ${f(100 * baseHits / n)}% accuracy`);
console.log(`Flat total ${f(baseTotal)}: ${f(test.reduce((s,t)=>s+Math.abs(baseTotal-t.at),0)/n)} MAE`);
console.log('===============================================\n');

const board = [...ratings.entries()]
  .filter(([, r]) => r.n > 5)
  .map(([name, r]) => ({ name, net: r.ortg - r.drtg, o: r.ortg, d: r.drtg, p: r.pace }))
  .sort((x, y) => y.net - x.net);

console.log('CURRENT RATINGS (sanity check — best to worst):');
board.slice(0, 6).forEach((t, i) =>
  console.log(`  ${i + 1}. ${t.name.padEnd(24)} net ${t.net >= 0 ? '+' : ''}${f(t.net)}`));
console.log('  ...');
board.slice(-3).forEach((t, i) =>
  console.log(`  ${board.length - 2 + i}. ${t.name.padEnd(24)} net ${t.net >= 0 ? '+' : ''}${f(t.net)}`));

fs.writeFileSync('./data/model.json', JSON.stringify({
  alpha: ALPHA,
  leagueAvg: league,
  marginFit: mFit,
  totalFit: tFit,
  asOf: games[games.length - 1].date,
  ratings: Object.fromEntries(ratings)
}, null, 2));
console.log('\nFitted model + live ratings written to data/model.json');
