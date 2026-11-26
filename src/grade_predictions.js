const fs = require('fs');

const PREDICTIONS = './data/prediction_log.json';
const RESULTS = './data/games.json';
const OUTPUT = './data/graded_predictions.json';

if (!fs.existsSync(PREDICTIONS) || !fs.existsSync(RESULTS)) {
  console.error('Missing prediction_log.json or games.json');
  process.exit(1);
}

const predictions = JSON.parse(fs.readFileSync(PREDICTIONS, 'utf8'));
const games = JSON.parse(fs.readFileSync(RESULTS, 'utf8'));

// Index actual results by NBA game ID
const results = new Map();

for (const g of games) {
  if (
    g.game_id &&
    Number.isFinite(Number(g.home_pts)) &&
    Number.isFinite(Number(g.away_pts))
  ) {
    results.set(String(g.game_id), g);
  }
}

let graded = 0;
let pending = 0;

// Preserve existing records if the script is run again
let existing = [];

if (fs.existsSync(OUTPUT)) {
  existing = JSON.parse(fs.readFileSync(OUTPUT, 'utf8'));
}

const previous = new Map(
  existing.map(p => [String(p.game_id), p])
);

const output = predictions.map(p => {
  const actual = results.get(String(p.game_id));

  if (!actual) {
    pending++;
    return {
      ...p,
      graded: false,
      grade_status: 'pending'
    };
  }

  const hp = Number(p.home_pts);
  const ap = Number(p.away_pts);

  const ah = Number(actual.home_pts);
  const aa = Number(actual.away_pts);

  if (![hp, ap, ah, aa].every(Number.isFinite)) {
    pending++;
    return {
      ...p,
      graded: false,
      grade_status: 'invalid_prediction_or_result'
    };
  }

  const predictedTotal = hp + ap;
  const actualTotal = ah + aa;

  const predictedMargin = hp - ap;
  const actualMargin = ah - aa;

  const predictedHomeWin = predictedMargin > 0;
  const actualHomeWin = actualMargin > 0;

  const old = previous.get(String(p.game_id)) || {};

  graded++;

  return {
    ...p,

    // Actual results
    actual_home_pts: ah,
    actual_away_pts: aa,
    actual_total: actualTotal,
    actual_margin: actualMargin,

    // Prediction errors
    home_score_error: +(hp - ah).toFixed(2),
    away_score_error: +(ap - aa).toFixed(2),
    total_error: +(predictedTotal - actualTotal).toFixed(2),
    margin_error: +(predictedMargin - actualMargin).toFixed(2),

    // Winner evaluation
    predicted_winner: predictedHomeWin
      ? p.home
      : p.away,

    actual_winner: actualHomeWin
      ? actual.home
      : actual.away,

    winner_correct: predictedHomeWin === actualHomeWin,

    // Probability evaluation
    home_win_probability: p.home_win,
    brier_score:
      Number.isFinite(Number(p.home_win))
        ? +((Number(p.home_win) - (actualHomeWin ? 1 : 0)) ** 2).toFixed(6)
        : null,

    graded: true,
    grade_status: 'completed',
    graded_at: old.graded_at || new Date().toISOString()
  };
});

fs.writeFileSync(
  OUTPUT,
  JSON.stringify(output, null, 2)
);

// Performance summary
const completed = output.filter(p => p.graded);

function mae(field) {
  if (!completed.length) return 'N/A';

  return (
    completed.reduce(
      (sum, p) => sum + Math.abs(p[field]),
      0
    ) / completed.length
  ).toFixed(2);
}

const winnerHits = completed.filter(p => p.winner_correct).length;

const brierRecords = completed.filter(
  p => Number.isFinite(p.brier_score)
);

const brier = brierRecords.length
  ? (
      brierRecords.reduce((s, p) => s + p.brier_score, 0) /
      brierRecords.length
    ).toFixed(4)
  : 'N/A';

console.log('\n======================================');
console.log('       NBA PREDICTION GRADING');
console.log('======================================');

console.log(`Total predictions : ${predictions.length}`);
console.log(`Graded games      : ${graded}`);
console.log(`Pending games     : ${pending}`);

console.log('--------------------------------------');
console.log(`Home points MAE   : ${mae('home_score_error')}`);
console.log(`Away points MAE   : ${mae('away_score_error')}`);
console.log(`Total points MAE  : ${mae('total_error')}`);
console.log(`Margin MAE        : ${mae('margin_error')}`);

console.log('--------------------------------------');

console.log(
  `Winner accuracy  : ${
    graded
      ? (100 * winnerHits / graded).toFixed(2) + '%'
      : 'N/A'
  }`
);

console.log(`Brier score       : ${brier}`);

console.log('======================================');

console.log(`\nSaved results to ${OUTPUT}\n`);
