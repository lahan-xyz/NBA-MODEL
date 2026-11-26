const fs = require('fs');
const M = JSON.parse(fs.readFileSync('./data/model.json','utf8'));

function normCdf(z){
  const t=1/(1+0.2316419*Math.abs(z));
  const d=0.3989423*Math.exp(-z*z/2);
  const p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));
  return z>0?1-p:p;
}

const args=process.argv.slice(2).filter(a=>!a.startsWith('--'));
const hb2b=process.argv.includes('--hb2b')?1:0;
const ab2b=process.argv.includes('--ab2b')?1:0;
if(args.length<2){
  console.log('Usage: node predict.js "<Home>" "<Away>" [--hb2b] [--ab2b]');
  console.log('Teams:', Object.keys(M.ratings).join(', '));
  process.exit(0);
}
const h=M.ratings[args[0]], a=M.ratings[args[1]];
if(!h||!a){ console.error('Team not found. Names above.'); process.exit(1); }

const pace=(h.p+a.p)/2;
const hRaw=(h.o+a.d-M.leagueAvg)*pace/100;
const aRaw=(a.o+h.d-M.leagueAvg)*pace/100;
const margin=M.marginFit.a+M.marginFit.b*(hRaw-aRaw)+M.b2b*(ab2b-hb2b);
const total=M.totalFit.a+M.totalFit.b*(hRaw+aRaw);
const hp=(total+margin)/2, ap=(total-margin)/2;
const win=100*normCdf(margin/M.residSD);
const f=(x,d=1)=>x.toFixed(d);

console.log(`\n${args[0]} vs ${args[1]}   (ratings as of ${M.asOf})`);
console.log(`  pace ${f(pace)}   net ${f(h.o-h.d)} vs ${f(a.o-a.d)}`);
console.log(`  ${args[0]}  ${f(hp)} pts   ${f(win)}% win`);
console.log(`  ${args[1]}  ${f(ap)} pts   ${f(100-win)}% win`);
console.log(`  total ${f(total)}   spread ${args[0]} ${margin>0?'-':'+'}${f(Math.abs(margin))}`);
console.log(`  B2B: home ${hb2b} away ${ab2b}   +/-${f(M.residSD)} pts residual SD`);
if(!hb2b&&!ab2b) console.log('  (both assumed rested — pass --hb2b/--ab2b if not)');
console.log('');
