const fs = require('fs');

const FILE = './data/graded_predictions.json';

if (!fs.existsSync(FILE)) {
  console.error('Missing data/graded_predictions.json');
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(FILE, 'utf8'));
const games = data.filter(p => p.graded === true);

if (!games.length) {
  console.log('No graded predictions available yet.');
  process.exit(0);
}

function avg(values) {
  return values.length
    ? values.reduce((a, b) => a + b, 0) / values.length
    : null;
}

function validNumbers(values) {
  return values.filter(v => typeof v === 'number' && Number.isFinite(v));
}

function mae(field) {
  const values = games
    .map(p => Number(p[field]))
    .filter(Number.isFinite)
    .map(Math.abs);

  return avg(values);
}

function pct(value) {
  return value === null ? 'N/A' : `${(value * 100).toFixed(2)}%`;
}

function fmt(value) {
  return value === null ? 'N/A' : value.toFixed(2);
}

// Determine actual home-team outcome
function actualHomeWin(p) {
  if (typeof p.actual_winner === 'string') {
    return p.actual_winner === p.home ? 1 : 0;
  }

  const home = Number(p.actual_home_pts);
  const away = Number(p.actual_away_pts);

  if (!Number.isFinite(home) || !Number.isFinite(away) || home === away) {
    return null;
  }

  return home > away ? 1 : 0;
}

// Keep only records with valid probabilities and actual outcomes
const probabilityGames = games
  .map(p => ({
    prediction: Number(p.home_win_probability),
    actual: actualHomeWin(p)
  }))
  .filter(p =>
    Number.isFinite(p.prediction) &&
    p.prediction >= 0 &&
    p.prediction <= 1 &&
    p.actual !== null
  );

const winnerHits = games.filter(p => p.winner_correct).length;
const winnerAccuracy = games.length
  ? winnerHits / games.length
  : null;

const brier = avg(
  probabilityGames.map(p => (p.prediction - p.actual) ** 2)
);

// Calibration groups
const groups = [
  { label: '0–50%', min: 0.00, max: 0.50 },
  { label: '50–60%', min: 0.50, max: 0.60 },
  { label: '60–70%', min: 0.60, max: 0.70 },
  { label: '70–80%', min: 0.70, max: 0.80 },
  { label: '80–90%', min: 0.80, max: 0.90 },
  { label: '90–100%', min: 0.90, max: 1.01 }
];

console.log('\n======================================');
console.log('         NBA MODEL PERFORMANCE');
console.log('======================================');

console.log(`Predictions graded: ${games.length}`);
console.log(`Predictions pending: ${data.length - games.length}`);

console.log('\n--- SCORING ACCURACY ---');
console.log(`Home points MAE: ${fmt(mae('home_score_error'))}`);
console.log(`Away points MAE: ${fmt(mae('away_score_error'))}`);
console.log(`Total points MAE: ${fmt(mae('total_error'))}`);
console.log(`Margin MAE: ${fmt(mae('margin_error'))}`);

console.log('\n--- WINNER PREDICTION ---');
console.log(`Winner accuracy: ${pct(winnerAccuracy)}`);
console.log(`Correct: ${winnerHits}/${games.length}`);

console.log('\n--- PROBABILITY QUALITY ---');
console.log(`Valid probability records: ${probabilityGames.length}`);
console.log(`Brier score: ${fmt(brier)}`);

console.log('\n--- HOME-WIN CALIBRATION ---');
console.log(
  'Group       Games   Predicted   Actual      Gap'
);
console.log('------------------------------------------------');

for (const group of groups) {
  const selected = probabilityGames.filter(p =>
    p.prediction >= group.min &&
    p.prediction < group.max
  );

  if (!selected.length) continue;

  const predictedRate = avg(selected.map(p => p.prediction));
  const actualRate = avg(selected.map(p => p.actual));
  const gap = predictedRate - actualRate;

  console.log(
    `${group.label.padEnd(11)} ` +
    `${String(selected.length).padEnd(7)} ` +
    `${pct(predictedRate).padEnd(11)} ` +
    `${pct(actualRate).padEnd(11)} ` +
    `${gap >= 0 ? '+' : ''}${(gap * 100).toFixed(2)} pp`
  );
}

console.log('------------------------------------------------');

console.log('\nInterpretation:');
console.log('+ Gap = model overestimates home-win probability.');
console.log('- Gap = model underestimates home-win probability.');
console.log('A gap near zero indicates better calibration.');

console.log('\n======================================\n');
