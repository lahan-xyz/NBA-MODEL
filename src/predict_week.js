const fs = require('fs');
const M = JSON.parse(fs.readFileSync('./data/model.json','utf8'));
const F = JSON.parse(fs.readFileSync('./data/fixtures.json','utf8'));

function normCdf(z){
  const t=1/(1+0.2316419*Math.abs(z));
  const d=0.3989423*Math.exp(-z*z/2);
  const p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));
  return z>0?1-p:p;
}
const now = new Date().toISOString();
const games = [];
for (const g of F.games) {
  const h=M.ratings[g.home], a=M.ratings[g.away];
  if(!h||!a){ console.warn('skip, no rating:', g.home, g.away); continue; }
  const pace=(h.p+a.p)/2;
  const hRaw=(h.o+a.d-M.leagueAvg)*pace/100;
  const aRaw=(a.o+h.d-M.leagueAvg)*pace/100;
  const margin=M.marginFit.a*(g.neutral?0:1)+M.marginFit.b*(hRaw-aRaw)+M.b2b*((g.away_b2b||0)-(g.home_b2b||0));
  const total=M.totalFit.a+M.totalFit.b*(hRaw+aRaw);
  games.push({
    game_id:g.game_id, date:g.date, home:g.home, away:g.away,
    home_pts:+((total+margin)/2).toFixed(1), away_pts:+((total-margin)/2).toFixed(1),
    total:+total.toFixed(1), margin:+margin.toFixed(1),
    home_win:+normCdf(margin/M.residSD).toFixed(3),
    neutral:g.neutral?1:0, home_b2b:g.home_b2b||0, away_b2b:g.away_b2b||0,
    model_asOf:M.asOf, predicted_at:now
  });
}
games.sort((x,y)=> x.date.localeCompare(y.date) || y.home_win-x.home_win);

fs.writeFileSync('./data/predictions.json', JSON.stringify({
  generated_at:now, model_asOf:M.asOf, window:F.window, count:games.length, games
}, null, 2));

// lock the FIRST prediction per game so grading later is honest
const LOG='./data/prediction_log.json';
let log=[]; try{ log=JSON.parse(fs.readFileSync(LOG,'utf8')); }catch(e){}
const seen=new Set(log.map(r=>r.game_id));
let added=0;
for(const r of games) if(!seen.has(r.game_id)){ log.push(r); added++; }
fs.writeFileSync(LOG, JSON.stringify(log,null,2));

console.log(`\n${games.length} predictions -> data/predictions.json  (${added} newly locked)`);
console.log('date        matchup                                   pts        total  home%');
for(const g of games){
  const m=`${g.away} @ ${g.home}`.padEnd(40).slice(0,40);
  const s=`${g.away_pts}-${g.home_pts}`.padStart(11);
  const flag=(g.home_b2b||g.away_b2b)?` B2B${g.home_b2b?'H':''}${g.away_b2b?'A':''}`:'';
  console.log(`${g.date}  ${m} ${s}  ${String(g.total).padStart(6)}  ${(100*g.home_win).toFixed(0)}%${flag}`);
}
console.log('');
