'use strict';

const express = require('express');
const https   = require('https');
const path    = require('path');
const fs      = require('fs');

const app  = express();
const PORT = process.env.PORT || 3456;

// ─── PREDEFINED STOCK LISTS ───────────────────────────────────────────────────

// Bundled inside project — works both locally and on Vercel
const LISTS_DIR = path.join(__dirname, 'data', 'lists');

const LIST_CONFIG = [
  { file: 'NIFTY50list.csv',              name: 'Nifty 50',              symCol: 'Symbol' },
  { file: 'niftynext50.csv',              name: 'Nifty Next 50',         symCol: 'Symbol' },
  { file: 'nifty100.csv',                 name: 'Nifty 100',             symCol: 'Symbol' },
  { file: 'niftymidcapselect.csv',        name: 'Nifty Midcap Select',   symCol: 'Symbol' },
  { file: 'niftymidcap50.csv',            name: 'Nifty Midcap 50',       symCol: 'Symbol' },
  { file: 'niftymidcap150.csv',           name: 'Nifty Midcap 150',      symCol: 'Symbol' },
  { file: 'niftysmallcap250.csv',         name: 'Nifty Smallcap 250',    symCol: 'Symbol' },
  { file: 'niftylargemidcap250.csv',      name: 'Nifty LargeMidcap 250', symCol: 'Symbol' },
  { file: 'niftymidsmallcap400.csv',      name: 'Nifty MidSmallcap 400', symCol: 'Symbol' },
  { file: 'niftymicrocap250.csv',         name: 'Nifty Microcap 250',    symCol: 'Symbol' },
  { file: 'NIFTY500list.csv',             name: 'Nifty 500',             symCol: 'Symbol' },
  { file: 'niftytotalmarket.csv',         name: 'Nifty Total Market',    symCol: 'Symbol' },
  { file: 'NIFTY_Complete_Equity_List.csv', name: 'NSE Complete Equity', symCol: 'Symbol' },
  { file: 'EQUITY NSE FULL.csv',          name: 'NSE Full (All Equity)', symCol: 'Symbol' },
];

const STOCK_LISTS = {};

function parseCSVSymbols(filepath, symCol) {
  const raw   = fs.readFileSync(filepath, 'utf8');
  const lines = raw.replace(/\r/g, '').trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.replace(/"/g, '').trim());
  let idx = headers.findIndex(h => h.toLowerCase() === symCol.toLowerCase());
  if (idx === -1) idx = 0;
  return lines.slice(1).map(line => {
    const cols = line.split(',');
    return (cols[idx] || '').replace(/"/g, '').trim().replace(/\.NS$/i, '');
  }).filter(s => s && s.length >= 2 && s.length <= 20 && /^[A-Z0-9&-]+$/.test(s));
}

function loadStockLists() {
  console.log('\n  Loading stock lists…');
  for (const cfg of LIST_CONFIG) {
    try {
      const syms = parseCSVSymbols(path.join(LISTS_DIR, cfg.file), cfg.symCol);
      STOCK_LISTS[cfg.name] = syms;
      console.log(`    ✓ ${cfg.name.padEnd(26)} ${syms.length} stocks`);
    } catch (e) {
      console.warn(`    ✗ ${cfg.file}: ${e.message}`);
    }
  }
  console.log('');
}

loadStockLists();

// ─── NSE SYMBOLS (fallback) ───────────────────────────────────────────────────

const NSE_SYMBOLS = [
  'ADANIENT','ADANIPORTS','APOLLOHOSP','ASIANPAINT','AXISBANK',
  'BAJAJ-AUTO','BAJAJFINSV','BAJFINANCE','BHARTIARTL','BPCL',
  'BRITANNIA','CIPLA','COALINDIA','DIVISLAB','DRREDDY',
  'EICHERMOT','GRASIM','HCLTECH','HDFCBANK','HDFCLIFE',
  'HEROMOTOCO','HINDALCO','HINDUNILVR','ICICIBANK','INDUSINDBK',
  'INFY','ITC','JSWSTEEL','KOTAKBANK','LT',
  'M&M','MARUTI','NESTLEIND','NTPC','ONGC',
  'POWERGRID','RELIANCE','SBIN','SBILIFE','SUNPHARMA',
  'TATAMOTORS','TATASTEEL','TCS','TATACONSUM','TECHM',
  'TITAN','ULTRACEMCO','UPL','WIPRO','LTIM',
  'ZOMATO','HAL','IRCTC','NAUKRI','DMART',
  'PIDILITIND','HAVELLS','DABUR','MARICO','COLPAL',
  'BERGEPAINT','GODREJCP','VOLTAS','AMBUJACEM','SHREECEM',
  'CANBK','PNB','BANKBARODA','FEDERALBNK','IDFCFIRSTB',
  'AUROPHARMA','TORNTPHARM','LUPIN','BIOCON','ALKEM',
  'MPHASIS','COFORGE','PERSISTENT','TRENT','POLYCAB',
  'ASTRAL','SIEMENS','ABB','DIXON','PAYTM'
];

// ─── CACHE ────────────────────────────────────────────────────────────────────

const cache = new Map();
const CACHE_TTL      = 5  * 60 * 1000;   // 5 min for live data
const CACHE_TTL_LONG = 30 * 60 * 1000;   // 30 min for 1y history

function fromCache(key, ttl = CACHE_TTL) {
  const e = cache.get(key);
  return (e && Date.now() - e.ts < ttl) ? e.data : null;
}
function toCache(key, data) { cache.set(key, { data, ts: Date.now() }); }

// ─── NSE INDIA DATA SOURCE ────────────────────────────────────────────────────

let nseSessionCookies = '';
let nseSessionTime    = 0;
let nseRefreshPromise = null;            // mutex: prevents concurrent double-refresh
const NSE_SESSION_TTL = 12 * 60 * 1000; // 12 min — NSE sessions expire fast

const NSE_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36';

async function refreshNSESession() {
  // Hit homepage first to pick up session cookies, then hit equity page
  const pages = ['/', '/get-quotes/equity?symbol=SBIN'];
  let cookies = {};

  for (const pagePath of pages) {
    await new Promise((resolve) => {
      const req = https.request({
        hostname: 'www.nseindia.com',
        path: pagePath,
        method: 'GET',
        headers: {
          'User-Agent': NSE_UA,
          'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.9',
          'Connection': 'keep-alive',
          ...(Object.keys(cookies).length ? { Cookie: Object.entries(cookies).map(([k,v]) => `${k}=${v}`).join('; ') } : {})
        }
      }, res => {
        for (const c of (res.headers['set-cookie'] || [])) {
          const [kv] = c.split(';');
          const [k, v=''] = kv.split('=');
          if (k.trim()) cookies[k.trim()] = v.trim();
        }
        res.resume();
        resolve();
      });
      req.on('error', () => resolve());
      req.setTimeout(10000, () => { req.destroy(); resolve(); });
      req.end();
    });
    await new Promise(r => setTimeout(r, 1200)); // NSE needs a brief pause between calls
  }

  nseSessionCookies = Object.entries(cookies).map(([k,v]) => `${k}=${v}`).join('; ');
  nseSessionTime    = Date.now();
}

async function nseApiGet(apiPath) {
  if (!nseSessionCookies || Date.now() - nseSessionTime > NSE_SESSION_TTL) {
    if (!nseRefreshPromise) {
      nseRefreshPromise = refreshNSESession().finally(() => { nseRefreshPromise = null; });
    }
    await nseRefreshPromise;
  }

  const doRequest = (cookies) => new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'www.nseindia.com',
      path: apiPath,
      method: 'GET',
      headers: {
        'User-Agent':       NSE_UA,
        'Accept':           'application/json, text/plain, */*',
        'Accept-Language':  'en-US,en;q=0.9',
        'Referer':          'https://www.nseindia.com/',
        'X-Requested-With': 'XMLHttpRequest',
        'Connection':       'keep-alive',
        'Cookie':           cookies
      }
    }, res => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', c => { body += c; });
      res.on('end', () => {
        if (res.statusCode === 401 || res.statusCode === 403) {
          return reject(new Error(`NSE_SESSION_EXPIRED:${res.statusCode}`));
        }
        if (res.statusCode >= 400) return reject(new Error(`NSE HTTP ${res.statusCode}`));
        try { resolve(JSON.parse(body)); }
        catch(e) { reject(new Error('NSE JSON parse failed')); }
      });
    });
    req.on('error', reject);
    req.setTimeout(15000, () => req.destroy(new Error('NSE timeout')));
    req.end();
  });

  try {
    return await doRequest(nseSessionCookies);
  } catch(e) {
    if (e.message.startsWith('NSE_SESSION_EXPIRED')) {
      // Refresh session once and retry
      await refreshNSESession();
      return await doRequest(nseSessionCookies);
    }
    throw e;
  }
}

// Format date DD-MM-YYYY for NSE API
function nseDateFmt(d) {
  return `${String(d.getDate()).padStart(2,'0')}-${String(d.getMonth()+1).padStart(2,'0')}-${d.getFullYear()}`;
}

// Fetch NSE historical daily OHLCV bars, chunked by 90 days to stay within API limits
async function fetchNSEDailyBars(symbol, days = 365) {
  const cacheKey = `nse:hist:${symbol}:${days}`;
  const cached   = fromCache(cacheKey, CACHE_TTL_LONG);
  if (cached) return cached;

  const to   = new Date();
  const from = new Date(to.getTime() - days * 86400000);

  // Build 90-day chunks oldest-first
  const chunks = [];
  let end = new Date(to);
  while (end > from) {
    const start = new Date(Math.max(end.getTime() - 90 * 86400000, from.getTime()));
    chunks.unshift({ from: nseDateFmt(start), to: nseDateFmt(end) });
    end = new Date(start.getTime() - 86400000);
  }

  const allBars = [];
  for (const chunk of chunks) {
    const series = encodeURIComponent('["EQ"]');
    const path   = `/api/historical/cm/equity?symbol=${encodeURIComponent(symbol)}&series=${series}&from=${chunk.from}&to=${chunk.to}`;
    try {
      const data = await nseApiGet(path);
      for (const r of (data.data || [])) {
        // NSE timestamp may be "2026-08-28" or "28-Aug-2026"
        const ts = Date.parse(r.CH_TIMESTAMP);
        if (isNaN(ts)) continue;
        const open  = parseFloat(r.CH_OPENING_PRICE);
        const high  = parseFloat(r.CH_TRADE_HIGH_PRICE);
        const low   = parseFloat(r.CH_TRADE_LOW_PRICE);
        const close = parseFloat(r.CH_CLOSING_PRICE);
        if (!close || !high || !low) continue;
        allBars.push({ time: ts, open, high, low, close, volume: parseInt(r.CH_TOT_TRADED_QTY) || 0 });
      }
    } catch(e) {
      console.warn(`[NSE] history chunk ${chunk.from}–${chunk.to} for ${symbol}: ${e.message}`);
    }
    await new Promise(r => setTimeout(r, 400)); // rate-limit between chunks
  }

  const bars = allBars
    .sort((a, b) => a.time - b.time)
    .filter((b, i, arr) => i === 0 || b.time !== arr[i-1].time);

  if (bars.length > 0) toCache(cacheKey, bars);
  return bars;
}

// Fetch today's live quote from NSE (3-5 min delayed, but better than Yahoo 15-20 min)
async function fetchNSEQuoteBar(symbol) {
  const data  = await nseApiGet(`/api/quote-equity?symbol=${encodeURIComponent(symbol)}`);
  const p     = data.priceInfo;
  if (!p || !p.lastPrice) throw new Error('NSE quote: missing priceInfo');
  const now   = Date.now();
  return {
    time:   now,
    open:   p.open         || p.previousClose,
    high:   p.intraDayHighLow?.max || p.lastPrice,
    low:    p.intraDayHighLow?.min || p.lastPrice,
    close:  p.lastPrice,
    volume: data.marketDeptOrderBook?.totalSellQuantity || 0
  };
}

// ─── HTTP HELPER ──────────────────────────────────────────────────────────────

function httpsGet(url) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = https.request({
      hostname: u.hostname,
      path: u.pathname + u.search,
      method: 'GET',
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9'
      }
    }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', c => { body += c; });
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve(body);
        else reject(new Error(`HTTP ${res.statusCode}`));
      });
    });
    req.on('error', reject);
    req.setTimeout(12000, () => req.destroy(new Error('Timeout')));
    req.end();
  });
}

async function fetchYahoo(symbol, interval, range) {
  const yahooSym = symbol + '.NS';
  const key = `${yahooSym}:${interval}:${range}`;
  const cached = fromCache(key);
  if (cached) return cached;
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(yahooSym)}?interval=${interval}&range=${range}&includePrePost=false`;
  const body = await httpsGet(url);
  const data = JSON.parse(body);
  if (data.chart?.error) throw new Error(data.chart.error.description || 'Yahoo error');
  toCache(key, data);
  return data;
}

// Fetch index/VIX symbols (no .NS suffix)
async function fetchYahooIndex(symbol, interval, range) {
  const key = `${symbol}:${interval}:${range}`;
  const ttl = range === '2y' ? CACHE_TTL_LONG : CACHE_TTL;
  const cached = fromCache(key, ttl);
  if (cached) return cached;
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=${interval}&range=${range}&includePrePost=false`;
  const body = await httpsGet(url);
  const data = JSON.parse(body);
  if (data.chart?.error) throw new Error(data.chart.error.description || 'Yahoo error');
  toCache(key, data);
  return data;
}

function extractBars(ydata) {
  try {
    const r = ydata.chart.result[0];
    if (!r || !r.timestamp) return [];
    const q = r.indicators.quote[0];
    return r.timestamp.map((ts, i) => ({
      time:   ts * 1000,
      open:   q.open[i]   ?? null,
      high:   q.high[i]   ?? null,
      low:    q.low[i]    ?? null,
      close:  q.close[i]  ?? null,
      volume: q.volume[i] ?? 0
    })).filter(b => b.high != null && b.low != null && b.close != null);
  } catch { return []; }
}

// ─── VIX + NIFTY GLOBAL MAPS ─────────────────────────────────────────────────

let vixMap        = {};   // { 'YYYY-MM-DD': vixClose }
let niftyMap      = {};   // { 'YYYY-MM-DD': niftyClose }
let niftyEma20Map = {};   // { 'YYYY-MM-DD': ema20 }
let niftyBarsCache= [];   // raw OHLCV for HMM regime computation

const mlEngine = require('./ml_engine');

function dateStr(tsMs) {
  return new Date(tsMs).toISOString().slice(0, 10);
}

async function loadIndexData() {
  try {
    const vixData = await fetchYahooIndex('^INDIAVIX', '1d', '2y');
    const vixBars = extractBars(vixData);
    vixMap = {};
    for (const b of vixBars) { if (b.close) vixMap[dateStr(b.time)] = b.close; }
    console.log(`  India VIX loaded: ${Object.keys(vixMap).length} days`);
  } catch (e) {
    console.warn(`  VIX load failed: ${e.message}`);
  }

  try {
    const nData  = await fetchYahooIndex('^NSEI', '1d', '2y');
    const nBars  = extractBars(nData);
    niftyMap = {};
    niftyBarsCache = nBars.filter(b => b.close).map(b => ({
      date: dateStr(b.time), close: b.close,
      high: b.high || b.close, low: b.low || b.close, volume: b.volume || 0
    }));
    for (const b of nBars) { if (b.close) niftyMap[dateStr(b.time)] = b.close; }

    // Compute 50-EMA series over Nifty (more stable regime filter than 20-EMA)
    const dates  = Object.keys(niftyMap).sort();
    const closes = dates.map(d => niftyMap[d]);
    const EMA_PERIOD = 50;
    const k = 2 / (EMA_PERIOD + 1);
    // Seed with SMA of first 50 closes for precision
    const seedLen = Math.min(EMA_PERIOD, closes.length);
    let ema = closes.slice(0, seedLen).reduce((a, b) => a + b, 0) / seedLen;
    niftyEma20Map = {};
    for (let i = 0; i < dates.length; i++) {
      ema = closes[i] * k + ema * (1 - k);
      niftyEma20Map[dates[i]] = ema;
    }
    console.log(`  Nifty 50 loaded:  ${dates.length} days`);
  } catch (e) {
    console.warn(`  Nifty load failed: ${e.message}`);
  }
}

// Load at startup; refresh every 6 hours
mlEngine.init();
loadIndexData().then(() => {
  // Compute initial HMM regime from Nifty bars
  if (niftyBarsCache.length >= 10) {
    const r = mlEngine.computeRegime(niftyBarsCache);
    console.log(`  ML Regime: ${r.regime} (score=${r.score})`);
  }
  setInterval(() => {
    loadIndexData().then(() => {
      if (niftyBarsCache.length >= 10) mlEngine.computeRegime(niftyBarsCache);
    });
  }, 6 * 60 * 60 * 1000);
});

// Latest available values (for today when market is open)
function latestVix() {
  const keys = Object.keys(vixMap).sort();
  return keys.length ? vixMap[keys[keys.length - 1]] : 0;
}
function latestNiftyAndEma() {
  const keys = Object.keys(niftyMap).sort();
  if (!keys.length) return { close: 0, ema20: 0 };
  const last = keys[keys.length - 1];
  return { close: niftyMap[last], ema20: niftyEma20Map[last] || 0 };
}

// ─── OOS SHARPE WEIGHTS (from backtest_v3 results) ───────────────────────────

const RULE_SHARPE = {
  rule1: 2.86, rule2: 4.75, rule3: 0.44,  rule4: 0.0,
  rule5: 3.24, rule6: 0.0,  rule7: 6.19,  rule8: 2.44,
  rule9: 4.45, rule10: 2.89, rule11: 5.27
};
// NOTE: R4 Sharpe=0.0 (neutral), R6 Sharpe=0.0 (was −1.61 OOS — enabled by user request, treat with caution)
const CONFLUENCE_THRESHOLD = 4.5;  // raised 3.5→4.5: only high-quality rule combos
const BREAKOUT_RULES       = new Set(['rule3', 'rule8']);
const MEAN_REVERSION_RULES = new Set(['rule9', 'rule10']);  // skip directional gates

// Per-rule optimal exit params from MFE/MAE 85th/60th pct backtest
const MFE_MAE_PARAMS = {
  rule1:  { optTargetPct: 2.687, optStopPct: 4.037 },
  rule2:  { optTargetPct: 2.525, optStopPct: 3.744 },
  rule3:  { optTargetPct: 2.537, optStopPct: 3.786 },
  rule4:  { optTargetPct: 2.500, optStopPct: 3.500 },
  rule5:  { optTargetPct: 2.609, optStopPct: 3.978 },
  rule6:  { optTargetPct: 3.033, optStopPct: 4.659 },
  rule7:  { optTargetPct: 2.526, optStopPct: 3.281 },
  rule8:  { optTargetPct: 2.671, optStopPct: 4.138 },
  rule9:  { optTargetPct: 2.000, optStopPct: 2.500 },
  rule10: { optTargetPct: 2.404, optStopPct: 3.174 },
  rule11: { optTargetPct: 2.767, optStopPct: 3.929 }
};

// ─── XGB PREDICTION CLIENT ────────────────────────────────────────────────────

const XGB_URL = 'http://127.0.0.1:5001/predict';
let xgbAvailable = false;

(async () => {
  try {
    const r = await fetch('http://127.0.0.1:5001/health');
    if (r.ok) { xgbAvailable = true; console.log('  XGBoost server: connected'); }
  } catch { console.log('  XGBoost server: not running (predictions disabled)'); }
})();

async function xgbPredict(featuresList) {
  if (!xgbAvailable) return null;
  try {
    const r = await fetch(XGB_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(featuresList),
      signal: AbortSignal.timeout(500)
    });
    if (!r.ok) return null;
    return (await r.json()).predictions || null;
  } catch { return null; }
}

// ─── QUALITY GATE INDICATORS ──────────────────────────────────────────────────

function calcEMA(arr, span) {
  if (!arr.length) return 0;
  const k = 2 / (span + 1);
  // Seed with SMA of first min(span,n) values — far more accurate than arr[0]
  const seedLen = Math.min(span, arr.length);
  let ema = arr.slice(0, seedLen).reduce((a, b) => a + b, 0) / seedLen;
  for (let i = seedLen; i < arr.length; i++) ema = arr[i] * k + ema * (1 - k);
  return ema;
}

function calcEMASeries(arr, span) {
  if (!arr.length) return [];
  const k = 2 / (span + 1);
  const seedLen = Math.min(span, arr.length);
  let ema = arr.slice(0, seedLen).reduce((a, b) => a + b, 0) / seedLen;
  const out = [];
  for (let i = 0; i < seedLen; i++) out.push(ema);  // fill seed region with SMA value
  for (let i = seedLen; i < arr.length; i++) {
    ema = arr[i] * k + ema * (1 - k);
    out.push(ema);
  }
  return out;
}

function calcRSI(closes, period = 14) {
  if (closes.length < period + 2) return 50;
  // Correct Wilder RSI: seed on first `period` diffs, then smooth remainder
  let ag = 0, al = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) ag += d; else al -= d;
  }
  ag /= period; al /= period;
  for (let i = period + 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    ag = (ag * (period - 1) + Math.max(d,  0)) / period;
    al = (al * (period - 1) + Math.max(-d, 0)) / period;
  }
  return al === 0 ? 100 : 100 - 100 / (1 + ag / al);
}

// RSI series — last N+1 values for divergence check
function calcRSISeries(closes, period = 14, lookback = 10) {
  const need = period + lookback + 1;
  const src  = closes.length >= need ? closes.slice(-need) : closes;
  const out  = [];
  if (src.length < period + 2) return out;
  let ag = 0, al = 0;
  for (let i = 1; i <= period; i++) {
    const d = src[i] - src[i-1];
    if (d > 0) ag += d; else al -= d;
  }
  ag /= period; al /= period;
  out.push(al === 0 ? 100 : 100 - 100 / (1 + ag / al));
  for (let i = period + 1; i < src.length; i++) {
    const d = src[i] - src[i-1];
    ag = (ag * (period-1) + Math.max(d,  0)) / period;
    al = (al * (period-1) + Math.max(-d, 0)) / period;
    out.push(al === 0 ? 100 : 100 - 100 / (1 + ag / al));
  }
  return out;
}

function rsiDivergence(closes, rsiVals, window = 5) {
  const n = closes.length, m = rsiVals.length;
  if (n < window + 2 || m < window + 2) return null;
  const pc = closes.slice(-window-1, -1), pr = rsiVals.slice(-window-1, -1);
  const curC = closes[n-1], curR = rsiVals[m-1];
  if (curC > Math.max(...pc) && curR < Math.max(...pr)) return 'bearish';
  if (curC < Math.min(...pc) && curR > Math.min(...pr)) return 'bullish';
  return null;
}

function calcATRRank(highs, lows, closes, period = 14, window = 252) {
  const n = highs.length;
  if (n < period + 2) return 0.5;
  const trs = [];
  for (let i = 1; i < n; i++)
    trs.push(Math.max(highs[i]-lows[i], Math.abs(highs[i]-closes[i-1]), Math.abs(lows[i]-closes[i-1])));
  let atr = trs.slice(0, period).reduce((a,b) => a+b, 0) / period;
  const atrSeries = [];
  for (let i = period; i < trs.length; i++) {
    atr = (atr*(period-1) + trs[i]) / period;
    atrSeries.push(atr);
  }
  if (!atrSeries.length) return 0.5;
  const curAtr = atrSeries[atrSeries.length - 1];
  const win = atrSeries.slice(-window).filter(v => v > 0);
  return win.length ? win.filter(v => v <= curAtr).length / win.length : 0.5;
}

// Volume Profile Point of Control — price bin with highest cumulative volume
function calcVolumePOC(bars, lookback = 20, bins = 50) {
  const recent = bars.slice(-lookback);
  if (recent.length < 5) return null;
  const lo = Math.min(...recent.map(b => b.low));
  const hi = Math.max(...recent.map(b => b.high));
  if (hi <= lo) return null;
  const binSize = (hi - lo) / bins;
  const vol = new Float64Array(bins);
  for (const b of recent) {
    const span = b.high - b.low || 1;
    for (let i = 0; i < bins; i++) {
      const bLo = lo + i * binSize, bHi = bLo + binSize;
      const overlap = Math.max(0, Math.min(b.high, bHi) - Math.max(b.low, bLo));
      vol[i] += (b.volume || 0) * overlap / span;
    }
  }
  let maxV = 0, pocBin = 0;
  for (let i = 0; i < bins; i++) if (vol[i] > maxV) { maxV = vol[i]; pocBin = i; }
  return lo + (pocBin + 0.5) * binSize;
}

function calcADX(highs, lows, closes, period = 14) {
  const n = highs.length;
  if (n < 2 * period + 2) return 0;
  const dmP = [], dmM = [], tr = [];
  for (let i = 1; i < n; i++) {
    const up = highs[i] - highs[i-1], dn = lows[i-1] - lows[i];
    dmP.push(up > dn && up > 0 ? up : 0);
    dmM.push(dn > up && dn > 0 ? dn : 0);
    tr.push(Math.max(highs[i]-lows[i], Math.abs(highs[i]-closes[i-1]), Math.abs(lows[i]-closes[i-1])));
  }
  const wilder = (arr, p) => {
    let s = arr.slice(0, p).reduce((a,b) => a+b, 0);
    const out = [s];
    for (let i = p; i < arr.length; i++) { s = s - s/p + arr[i]; out.push(s); }
    return out;
  };
  const sTR = wilder(tr, period), sP = wilder(dmP, period), sM = wilder(dmM, period);
  const DX = sTR.map((t, i) => {
    const dp = t > 0 ? 100*sP[i]/t : 0, dm = t > 0 ? 100*sM[i]/t : 0;
    return dp+dm > 0 ? 100*Math.abs(dp-dm)/(dp+dm) : 0;
  });
  let adx = DX.slice(0, period).reduce((a,b) => a+b, 0) / period;
  for (let i = period; i < DX.length; i++) adx = (adx*(period-1) + DX[i]) / period;
  return adx;
}

function kalmanVelocity(closes, Q = 0.001, R = 0.1) {
  if (closes.length < 2) return 0;
  let x = closes[0], P = 1.0, prevX = closes[0];
  for (let j = 0; j < closes.length; j++) {
    P += Q;
    const K = P / (P + R);
    const xNew = x + K * (closes[j] - x);
    P = (1 - K) * P;
    prevX = x;
    x = xNew;
  }
  return x - prevX;  // Kalman state velocity (smoothed→smoothed, not smoothed→raw)
}

function sgVelocity(closes, winSize = 11) {
  const n = Math.min(winSize, closes.length);
  const y = closes.slice(-n);
  const xm = (n-1)/2, ym = y.reduce((a,b) => a+b, 0)/n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i-xm)*(y[i]-ym); den += (i-xm)**2; }
  return den > 0 ? num/den : 0;
}

// ─── RULE DIRECTION ───────────────────────────────────────────────────────────

function getRuleDirection(rid, cpr, cam, currentClose, prevClose, periodHigh, periodLow) {
  if (rid === 'rule3')  return 1;
  if (rid === 'rule4')  return periodLow > cpr.upper ? 1 : -1;
  if (rid === 'rule6') {
    const safe = currentClose > 0 ? currentClose : 1;
    return Math.abs(currentClose - cam.r3) / safe < 0.005 ? -1 : 1;
  }
  if (rid === 'rule7')  return prevClose > cpr.upper ? 1 : -1;
  if (rid === 'rule8')  return currentClose > cpr.upper ? 1 : -1;
  if (rid === 'rule9')  return currentClose > cpr.pivot ? -1 : 1;
  if (rid === 'rule10') {
    return (cpr.upper > 0 && Math.abs(periodHigh - cpr.upper) / cpr.upper < 0.005) ? -1 : 1;
  }
  return currentClose >= cpr.pivot ? 1 : -1;
}

// ─── CPR / CAM / VWAP ────────────────────────────────────────────────────────

function calcCPR(H, L, C) {
  const pivot = (H + L + C) / 3;
  const bc    = (H + L) / 2;
  const tc    = (pivot - bc) + pivot;
  const lower = Math.min(tc, bc);
  const upper = Math.max(tc, bc);
  const width    = upper - lower;
  const widthPct = pivot > 0 ? (width / pivot) * 100 : 0;
  return { pivot, bc, tc, lower, upper, width, widthPct };
}

function calcCamarilla(H, L, C) {
  const r = H - L, f = 1.1;
  const raw = {
    r4: C+r*f/2, r3: C+r*f/4, r2: C+r*f/6, r1: C+r*f/12,
    s1: C-r*f/12, s2: C-r*f/6, s3: C-r*f/4, s4: C-r*f/2
  };
  const p = v => +v.toFixed(2);
  return { r4:p(raw.r4), r3:p(raw.r3), r2:p(raw.r2), r1:p(raw.r1),
           s1:p(raw.s1), s2:p(raw.s2), s3:p(raw.s3), s4:p(raw.s4) };
}

function calcVWAP(bars) {
  let pv = 0, vol = 0;
  return bars.map(b => {
    const tp = (b.high + b.low + b.close) / 3;
    pv += tp * b.volume; vol += b.volume;
    return vol > 0 ? pv / vol : tp;
  });
}

function aggregateBars(bars) {
  if (!bars.length) return null;
  return {
    high:   Math.max(...bars.map(b => b.high)),
    low:    Math.min(...bars.map(b => b.low)),
    close:  bars[bars.length - 1].close,
    open:   bars[0].open,
    volume: bars.reduce((s, b) => s + b.volume, 0),
    time:   bars[bars.length - 1].time
  };
}

// ─── RULES ENGINE ─────────────────────────────────────────────────────────────

const RULES = {
  rule1: {
    name: 'Camarilla S3 or R3 Inside CPR',
    desc: 'Camarilla S3 or R3 level falls within the CPR band — price compression near key pivot',
    color: '#8B5CF6',
    check({ cpr, cam }) {
      return (cam.s3 >= cpr.lower && cam.s3 <= cpr.upper) ||
             (cam.r3 >= cpr.lower && cam.r3 <= cpr.upper);
    }
  },
  rule2: {
    name: 'Narrow CPR',
    desc: 'CPR width below 0.5% — potential explosive move',
    color: '#06B6D4',
    check({ cpr, narrowThreshold }) { return cpr.widthPct < (narrowThreshold || 0.5); }
  },
  rule3: {
    name: 'Close Crossing Above TC',
    desc: 'Price crossed above Top Central Pivot — bullish breakout',
    color: '#10B981',
    check({ prevClose, currentClose, cpr }) {
      return prevClose != null && prevClose < cpr.upper && currentClose > cpr.upper;
    }
  },
  rule4: {
    name: 'Virgin CPR',
    desc: "Price has not yet entered CPR zone this period",
    color: '#F59E0B',
    check({ cpr, periodHigh, periodLow }) {
      return periodHigh < cpr.lower || periodLow > cpr.upper;
    }
  },
  rule5: {
    name: 'CPR + VWAP Confluence',
    desc: 'VWAP overlaps CPR band — double S/R zone',
    color: '#ec4899',
    check({ cpr, vwap }) {
      const margin = Math.max(cpr.width * 0.5, cpr.pivot * 0.002);
      return vwap >= cpr.lower - margin && vwap <= cpr.upper + margin;
    }
  },
  rule6: {
    name: 'Wide CPR + Cam Extreme',
    desc: 'Wide CPR (>0.7%) + price at Cam R3/S3 — mean-reversion fade',
    color: '#f97316',
    check({ cpr, cam, currentClose }) {
      const isWide = cpr.widthPct > 0.7;
      const safe   = currentClose > 0 ? currentClose : 1;
      return isWide && (Math.abs(currentClose-cam.r3)/safe < 0.005 ||
                        Math.abs(currentClose-cam.s3)/safe < 0.005);
    }
  },
  rule7: {
    name: 'CPR S/R Flip Retest',
    desc: 'Price broke CPR boundary and retesting from other side — precision entry',
    color: '#14b8a6',
    check({ prevClose, currentClose, cpr }) {
      const rs = prevClose > cpr.upper && currentClose > cpr.upper &&
                 (currentClose - cpr.upper) / cpr.upper < 0.012;
      const rr = prevClose < cpr.lower && currentClose < cpr.lower &&
                 (cpr.lower - currentClose) / cpr.lower < 0.012;
      return rs || rr;
    }
  },
  rule8: {
    name: 'Opening Range + CPR Aligned',
    desc: 'Early bars break in same direction as CPR — momentum confirmation',
    color: '#a855f7',
    check({ firstBars, currentClose, cpr }) {
      if (!firstBars || firstBars.length < 1) return false;
      const orHigh = Math.max(...firstBars.map(b => b.high));
      const orLow  = Math.min(...firstBars.map(b => b.low));
      return (currentClose > orHigh && currentClose > cpr.upper) ||
             (currentClose < orLow  && currentClose < cpr.lower);
    }
  },
  rule9: {
    name: 'Pivot Magnetic Pull',
    desc: 'Price >2% from CPR pivot — mean-reversion targeting pivot',
    color: '#eab308',
    check({ cpr, currentClose }) {
      return cpr.pivot > 0 && Math.abs(currentClose - cpr.pivot) / cpr.pivot > 0.02;
    }
  },
  rule10: {
    name: 'Price Testing CPR as S/R',
    desc: 'Period high testing CPR upper or low testing CPR lower — live S/R in action',
    color: '#22d3ee',
    check({ periodHigh, periodLow, cpr }) {
      return (cpr.upper > 0 && Math.abs(periodHigh-cpr.upper)/cpr.upper < 0.005) ||
             (cpr.lower > 0 && Math.abs(periodLow -cpr.lower)/cpr.lower < 0.005);
    }
  },
  rule11: {
    name: 'VWAP-to-TC Setup',
    desc: 'Price between VWAP and CPR boundary — breakout with VWAP as support',
    color: '#84cc16',
    check({ vwap, currentClose, cpr }) {
      return (currentClose > vwap && currentClose < cpr.upper) ||
             (currentClose < vwap && currentClose > cpr.lower);
    }
  }
};

function evaluateRules(ctx, activeRules) {
  const out = {};
  for (const id of activeRules) {
    if (RULES[id]) {
      try { out[id] = RULES[id].check(ctx); } catch { out[id] = false; }
    }
  }
  return out;
}

// ─── TIMEFRAME CONFIG ─────────────────────────────────────────────────────────

const TF = {
  '5m': {
    label: '5 Min', tvInterval: '5',
    getPrev: async (sym) => {
      const d = await fetchYahoo(sym, '1d', '5d');
      const bars = extractBars(d);
      if (bars.length < 2) throw new Error('No daily history');
      return bars[bars.length - 2];
    },
    getCurrent: async (sym) => {
      const d = await fetchYahoo(sym, '5m', '1d');
      return extractBars(d);
    }
  },
  '1d': {
    label: '1 Day', tvInterval: 'D',
    getPrev: async (sym) => {
      // Try NSE first (authoritative, unadjusted prices)
      try {
        const bars = await fetchNSEDailyBars(sym, 10);
        if (bars.length >= 1) {
          const todayMidnight = new Date(); todayMidnight.setHours(0,0,0,0);
          const lastTs = bars[bars.length - 1].time;
          // If bars[-1] is today's partial bar, yesterday = bars[-2]
          // If bars[-1] is yesterday (NSE hasn't pushed today yet), yesterday = bars[-1]
          if (lastTs >= todayMidnight.getTime()) {
            if (bars.length >= 2) return bars[bars.length - 2];
          } else {
            return bars[bars.length - 1];
          }
        }
      } catch(e) { console.warn(`[NSE] getPrev fallback for ${sym}: ${e.message}`); }
      // Fallback: Yahoo Finance
      const d = await fetchYahoo(sym, '1d', '5d');
      const bars = extractBars(d);
      if (bars.length < 1) throw new Error('No daily history');
      const todayMidnight2 = new Date(); todayMidnight2.setHours(0,0,0,0);
      const lastTs2 = bars[bars.length - 1].time;
      if (lastTs2 >= todayMidnight2.getTime()) {
        if (bars.length < 2) throw new Error('No daily history');
        return bars[bars.length - 2];
      }
      return bars[bars.length - 1];
    },
    getCurrent: async (sym) => {
      // Try NSE: historical bars + today's live quote bar
      try {
        const bars = await fetchNSEDailyBars(sym, 10);
        if (bars.length >= 1) {
          // Append today's live quote as the last bar (3-5 min delayed)
          try {
            const todayBar = await fetchNSEQuoteBar(sym);
            // Only append if today's bar isn't already in history
            const lastHistTs = bars[bars.length - 1].time;
            const todayMidnight = new Date(); todayMidnight.setHours(0,0,0,0);
            if (lastHistTs < todayMidnight.getTime()) {
              bars.push(todayBar);
            } else {
              // Update last bar's close/high/low with live data
              const last = bars[bars.length - 1];
              last.close  = todayBar.close;
              last.high   = Math.max(last.high, todayBar.high);
              last.low    = Math.min(last.low,  todayBar.low);
            }
          } catch { /* quote fetch failed — use historical close as-is */ }
          return bars.slice(-5);
        }
      } catch(e) { console.warn(`[NSE] getCurrent fallback for ${sym}: ${e.message}`); }
      // Fallback: Yahoo Finance
      const d = await fetchYahoo(sym, '1d', '1mo');
      return extractBars(d).slice(-5);
    }
  },
  '1w': {
    label: '1 Week', tvInterval: 'W',
    getPrev: async (sym) => {
      const d = await fetchYahoo(sym, '1wk', '1y');
      const bars = extractBars(d);
      if (bars.length < 5) return bars[bars.length - 2] || bars[0];
      return aggregateBars(bars.slice(-5, -1));
    },
    getCurrent: async (sym) => {
      const d = await fetchYahoo(sym, '1wk', '6mo');
      return extractBars(d).slice(-4);
    }
  },
  '1M': {
    label: '1 Month', tvInterval: 'M',
    getPrev: async (sym) => {
      const d = await fetchYahoo(sym, '1mo', '3y');
      const bars = extractBars(d);
      if (bars.length < 13) return bars[bars.length - 2] || bars[0];
      return aggregateBars(bars.slice(-13, -1));
    },
    getCurrent: async (sym) => {
      const d = await fetchYahoo(sym, '1mo', '2y');
      return extractBars(d).slice(-12);
    }
  }
};

// ─── SYMBOL PROCESSOR ─────────────────────────────────────────────────────────

async function processSymbol(symbol, timeframe, activeRules, opts = {}) {
  const tf = TF[timeframe] || TF['1d'];
  try {
    // Fetch current data + 1y history in parallel for quality gates
    // For daily timeframe: fetch 1y NSE history for quality gates (ATR/ADX/EMA/Kalman)
    // Other timeframes: Yahoo Finance (NSE doesn't have intraday/weekly API)
    const histFetch = (timeframe === '1d')
      ? fetchNSEDailyBars(symbol, 365).catch(() => fetchYahoo(symbol, '1d', '1y').catch(() => null))
      : fetchYahoo(symbol, '1d', '1y').catch(() => null);

    const [prevPeriod, currentBars, histDataRaw] = await Promise.all([
      tf.getPrev(symbol),
      tf.getCurrent(symbol),
      histFetch
    ]);
    // Normalize: NSE returns bars array directly; Yahoo returns raw ydata object
    const histData = Array.isArray(histDataRaw) ? { _nseBars: histDataRaw } : histDataRaw;

    if (!prevPeriod || !currentBars.length) throw new Error('Insufficient data');

    const last  = currentBars[currentBars.length - 1];
    const prev2 = currentBars.length >= 2 ? currentBars[currentBars.length - 2] : null;

    const cpr = calcCPR(prevPeriod.high, prevPeriod.low, prevPeriod.close);
    const cam = calcCamarilla(prevPeriod.high, prevPeriod.low, prevPeriod.close);
    const vwapArr = calcVWAP(currentBars);
    const vwap    = vwapArr[vwapArr.length - 1];

    const periodHigh = Math.max(...currentBars.map(b => b.high));
    const periodLow  = Math.min(...currentBars.map(b => b.low));
    const prevClose  = prev2?.close ?? prevPeriod.close;
    const change     = +((last.close - prevClose) / prevClose * 100).toFixed(2);

    // ── Quality gate computation ───────────────────────────────────────────────
    const histBars = histData
      ? (histData._nseBars ? histData._nseBars : extractBars(histData))
      : currentBars;
    const hCloses  = histBars.map(b => b.close);
    const hHighs   = histBars.map(b => b.high);
    const hLows    = histBars.map(b => b.low);
    const hVols    = histBars.map(b => b.volume || 0);

    // Volume Profile POC
    const poc        = calcVolumePOC(histBars, 20, 50);
    const pocDistPct = (poc && cpr.pivot > 0) ? Math.abs(poc - cpr.pivot) / cpr.pivot * 100 : 999;
    const pocAligned = pocDistPct < 0.8;   // POC within 0.8% of CPR pivot = strong confluence

    // Global quality gates (rule-agnostic)
    const adxVal    = histBars.length >= 30 ? calcADX(hHighs, hLows, hCloses) : 25;
    const atrPctVal = histBars.length >= 30 ? calcATRRank(hHighs, hLows, hCloses) : 0.5;
    const vol20avg  = hVols.length >= 20 ? hVols.slice(-20).reduce((a,b)=>a+b,0)/20
                                         : hVols.reduce((a,b)=>a+b,0)/(hVols.length||1) || 0;
    const lastVol   = hVols[hVols.length-1] || 0;
    // Fail-open when no volume data (vol20avg=0 means Yahoo returned no volume)
    const volRatio  = vol20avg > 0 ? lastVol / vol20avg : 1.5;
    // Volume trend: is recent momentum accelerating?
    const vol3avg   = hVols.length >= 3  ? hVols.slice(-3).reduce((a,b)=>a+b,0)/3   : 0;
    const vol10avg  = hVols.length >= 10 ? hVols.slice(-10).reduce((a,b)=>a+b,0)/10 : 0;

    // VIX gate
    const todayStr = new Date().toISOString().slice(0, 10);
    const vixVal   = vixMap[todayStr] ?? latestVix();
    const vixPass  = vixVal === 0 || vixVal < 18;   // fail-open if no data

    // Nifty 20-EMA gate (value stored; direction applied per rule below)
    const { close: niftyClose, ema20: niftyEma20 } = latestNiftyAndEma();

    // Direction-agnostic gates
    const dg = new Set(opts.disabledGates || []);

    const gateADX    = dg.has('adx')      ? true : adxVal   >= 25;           // raised 20→25: require confirmed trend
    const gateATRPct = dg.has('atrPct')   ? true : (atrPctVal >= 0.40 && atrPctVal <= 0.70);  // tightened: prime vol zone
    const gateVol    = dg.has('volSurge') ? true : volRatio >= 1.0;
    const gateVIX    = dg.has('vix')      ? true : vixPass;

    // Directional indicators (computed once, applied per rule)
    const ema200val = hCloses.length >= 20 ? calcEMA(hCloses, Math.min(200, hCloses.length)) : last.close;
    const kalVel    = hCloses.length >= 5  ? kalmanVelocity(hCloses.slice(-60), 0.005, 0.1)  : 0;  // Q=0.005: less lag
    const sgVel     = sgVelocity(hCloses.slice(-20));

    // RSI series for divergence
    const rsiSeries = calcRSISeries(hCloses, 14, 10);
    const rsiCur    = rsiSeries.length ? rsiSeries[rsiSeries.length - 1] : 50;
    const closesForDiv = hCloses.slice(-(rsiSeries.length));
    const rsiDiv    = rsiDivergence(closesForDiv, rsiSeries);

    // Weighted confluence
    const ctx = {
      cpr, cam, vwap, prevClose,
      currentClose: last.close,
      periodHigh, periodLow,
      prevPeriodHigh: prevPeriod.high,
      prevPeriodLow:  prevPeriod.low,
      periodOpen:  currentBars[0].open,
      firstBars:   currentBars.slice(0, 3),
      timeframe,
      narrowThreshold: opts.narrowThreshold || 0.5
    };

    const ruleResults  = evaluateRules(ctx, activeRules);
    const matchedRules = Object.entries(ruleResults).filter(([,v]) => v).map(([k]) => k);

    // ── Per-rule quality gate evaluation ──────────────────────────────────────
    const ruleGates      = {};
    const qualityPassedRules = [];
    const exitLevelsByRule   = {};

    for (const rid of matchedRules) {

      const direction    = getRuleDirection(rid, cpr, cam, last.close, prevClose, periodHigh, periodLow);
      const isMeanRev    = MEAN_REVERSION_RULES.has(rid);
      const isBreakout   = BREAKOUT_RULES.has(rid);

      // ② Directional gates — skip for mean-reversion rules
      // Raw directional values (for badge display)
      const rawEMA200 = direction === 1 ? last.close >= ema200val * 0.995 : last.close <= ema200val * 1.005; // tightened 1%→0.5%
      const rawKalman = direction === 1 ? kalVel >= 0 : kalVel <= 0;
      const rawSG     = direction === 1 ? sgVel  >= 0 : sgVel  <= 0;
      const rawNifty  = niftyEma20 <= 0 ? true
        : (direction === 1 ? niftyClose >= niftyEma20 * 0.998 : niftyClose <= niftyEma20 * 1.002);  // tightened 0.5%→0.2%

      // For badge: show each individual gate value
      const gateEMA200 = dg.has('ema200')   ? true : rawEMA200;
      const gateKalman = dg.has('kalman')   ? true : rawKalman;
      const gateSG     = dg.has('sgTrend')  ? true : rawSG;
      const gateNifty  = dg.has('niftyEma') ? true : rawNifty;

      // For allPass: ≥2 of 4 directional signals must agree (or rule is mean-rev)
      const dirScore = (dg.has('ema200')   || rawEMA200 ? 1 : 0)
                     + (dg.has('kalman')   || rawKalman  ? 1 : 0)
                     + (dg.has('sgTrend')  || rawSG      ? 1 : 0)
                     + (dg.has('niftyEma') || rawNifty   ? 1 : 0);
      const gateDir = isMeanRev ? true : dirScore >= 3;  // require 3-of-4 directional consensus

      // ③ RSI divergence — only breakout rules
      const gateRSIDiv = (isBreakout && !dg.has('rsiDiv'))
        ? !(direction === 1 && rsiDiv === 'bearish') && !(direction === -1 && rsiDiv === 'bullish')
        : true;

      // ④ POC — informational only; badge shows alignment, toggle enforces it
      const gatePOC = dg.has('poc') ? true : pocAligned;

      // ⑤ RSI at-entry filter: longs want RSI 40-75, shorts want 25-60; skip for mean-rev
      const gateRSIEntry = (isMeanRev || dg.has('rsiEntry'))
        ? true
        : (direction === 1 ? rsiCur >= 40 && rsiCur <= 75 : rsiCur >= 25 && rsiCur <= 60);

      // ⑥ Last candle body confirms direction
      const gateCandle = dg.has('candleConf')
        ? true
        : (direction === 1 ? last.close >= last.open : last.close <= last.open);

      // ⑦ Volume trend: recent 3-bar avg should be ≥90% of 10-bar avg (accumulation)
      const gateVolTrend = dg.has('volTrend')
        ? true
        : (vol10avg > 0 ? vol3avg / vol10avg >= 0.9 : true);

      const allPass = gateADX && gateATRPct && gateVol && gateVIX && gateDir && gateRSIDiv && gateRSIEntry && gateCandle && gateVolTrend;

      ruleGates[rid] = {
        direction,
        pass: allPass,
        isMeanRev,
        gates: {
          adx:     { value: +adxVal.toFixed(1),      pass: gateADX },
          atrPct:  { value: +atrPctVal.toFixed(2),   pass: gateATRPct },
          volSurge:{ value: +volRatio.toFixed(2),     pass: gateVol },
          vix:     { value: +vixVal.toFixed(2),       pass: gateVIX },
          ema200:  { value: +ema200val.toFixed(2),    pass: gateEMA200,  skipped: isMeanRev },
          kalman:  { value: +kalVel.toFixed(4),       pass: gateKalman,  skipped: isMeanRev },
          sg:      { value: +sgVel.toFixed(4),        pass: gateSG,      skipped: isMeanRev },
          niftyEma:{ niftyClose: +niftyClose.toFixed(0), ema20: +niftyEma20.toFixed(0), pass: gateNifty, skipped: isMeanRev },
          dir:     { score: dirScore, pass: gateDir,  skipped: isMeanRev },
          rsiDiv:   { divergence: rsiDiv,              pass: gateRSIDiv,   skipped: !isBreakout },
          rsiEntry: { value: +rsiCur.toFixed(1),      pass: gateRSIEntry, skipped: isMeanRev },
          candle:   { close: +last.close.toFixed(2), open: +last.open.toFixed(2), pass: gateCandle },
          volTrend: { vol3: +vol3avg.toFixed(0), vol10: +vol10avg.toFixed(0), pass: gateVolTrend },
          poc:      { value: poc ? +poc.toFixed(2) : null, distPct: +pocDistPct.toFixed(2), pass: pocAligned, info: true }
        }
      };

      if (allPass) {
        qualityPassedRules.push(rid);

        // ⑤ Optimal exit levels from MFE/MAE backtest
        const ep = MFE_MAE_PARAMS[rid] || { optTargetPct: 2.5, optStopPct: 3.5 };
        const entry = last.close;
        exitLevelsByRule[rid] = {
          direction,
          entry:      +entry.toFixed(2),
          t1CutPrice: +(entry * (1 - 0.012)).toFixed(2),        // -1.2% day-1 cut
          trailStop:  +(entry * (1 - 0.008)).toFixed(2),        // -0.8% trailing stop
          optTarget:  +(entry * (1 + ep.optTargetPct/100) * (direction === 1 ? 1 : -1) +
                        entry * (direction === -1 ? 2 : 0)).toFixed(2),
          targetPct:  ep.optTargetPct,
          stopPct:    ep.optStopPct,
          targetPrice: direction === 1
            ? +(entry * (1 + ep.optTargetPct / 100)).toFixed(2)
            : +(entry * (1 - ep.optTargetPct / 100)).toFixed(2),
          stopPrice: direction === 1
            ? +(entry * (1 - ep.optStopPct  / 100)).toFixed(2)
            : +(entry * (1 + ep.optStopPct  / 100)).toFixed(2)
        };
      }
    }

    // Confluence score (only quality-passed rules) + CPR width quality adjustment
    const cprWidthPct   = cpr.pivot > 0 ? Math.abs(cpr.tc - cpr.bc) / cpr.pivot * 100 : 0.5;
    const cprWidthBonus = cprWidthPct < 0.3 ? 0.5 : cprWidthPct > 1.0 ? -0.5 : 0;  // narrow=bonus, wide=penalty
    const confluenceScore = qualityPassedRules.reduce((s, r) => s + (RULE_SHARPE[r] || 0), 0) + cprWidthBonus;
    const confluencePass  = confluenceScore >= CONFLUENCE_THRESHOLD;

    // ── XGBoost predictions ────────────────────────────────────────────────────
    let predReturnByRule = {};
    let featureList = [];
    const rulesToPredict = qualityPassedRules.length > 0 ? qualityPassedRules : matchedRules;

    if (rulesToPredict.length > 0 && xgbAvailable) {
      // Use full historical arrays (365-bar) so features match training window sizes
      const dow      = new Date().getDay();

      // Shared cross-rule features computed once from histBars
      const hi52      = hHighs.length > 0 ? Math.max(...hHighs) : last.close;
      const lo52      = hLows.length  > 0 ? Math.min(...hLows)  : last.close;
      const dist_hi52 = last.close > 0 ? (hi52 - last.close) / last.close : 0;
      const dist_lo52 = last.close > 0 ? (last.close - lo52)  / last.close : 0;
      // vol_accel: 3-bar avg / 20-bar avg (vol3avg/vol20avg already computed above)
      const xVol20a   = vol20avg > 0 ? vol20avg : 1;
      const vol_accel = xVol20a > 0 ? vol3avg / xVol20a : 1;
      // RSI and sg_vel over full history so window matches training
      const xRsi14    = calcRSI(hCloses, 14);
      const xSgVel    = sgVelocity(hCloses.slice(-21), 21); // 21-bar window matches Python savgol training
      const xMom5     = hCloses.length >= 6
                          ? (hCloses[hCloses.length-1] - hCloses[hCloses.length-6]) / hCloses[hCloses.length-6]
                          : 0;
      const xMom3  = hCloses.length >= 4
                       ? (hCloses[hCloses.length-1] - hCloses[hCloses.length-4]) / hCloses[hCloses.length-4]
                       : 0;
      const xMom10 = hCloses.length >= 11
                       ? (hCloses[hCloses.length-1] - hCloses[hCloses.length-11]) / hCloses[hCloses.length-11]
                       : 0;
      const xMom20 = hCloses.length >= 21
                       ? (hCloses[hCloses.length-1] - hCloses[hCloses.length-21]) / hCloses[hCloses.length-21]
                       : 0;

      const gap_pct = prevClose > 0 ? (last.open - prevClose) / prevClose : 0;

      // atr_expansion: today ATR vs 10-day avg ATR
      const _atrToday = hHighs.length > 0 ? hHighs[hHighs.length-1] - hLows[hLows.length-1] : 0;
      let _atr10avg = _atrToday;
      if (hHighs.length >= 10) {
        const _atrs = hHighs.slice(-10).map((h, i) => h - hLows[hLows.length - 10 + i]);
        _atr10avg = _atrs.reduce((a, b) => a + b, 0) / 10;
      }
      const atr_expansion = _atr10avg > 0 ? _atrToday / _atr10avg : 1;

      // vol_trend_slope: normalized OLS slope over last 20 volumes
      let vol_trend_slope = 0;
      if (hVols.length >= 5) {
        const vSlice = hVols.slice(-Math.min(20, hVols.length));
        const n = vSlice.length;
        const sumX = n * (n - 1) / 2, sumX2 = n * (n - 1) * (2 * n - 1) / 6;
        const sumY = vSlice.reduce((a, b) => a + b, 0);
        const sumXY = vSlice.reduce((s, v, i) => s + i * v, 0);
        const denom = n * sumX2 - sumX * sumX;
        vol_trend_slope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom / (xVol20a || 1) : 0;
      }

      // days_since_52hi: bars-ago index of 52w high, normalized to trading-year
      const _hiMax = hHighs.length > 0 ? Math.max(...hHighs) : last.close;
      const _hi52idx = hHighs.lastIndexOf(_hiMax);
      const days_since_52hi = hHighs.length > 0 ? (hHighs.length - 1 - _hi52idx) / 252 : 0;

      // expiry_dist: fraction of 30-day window remaining to last Thursday of current month
      const _now = new Date();
      const _lastThurs = (() => {
        const d = new Date(_now.getFullYear(), _now.getMonth() + 1, 0);
        d.setDate(d.getDate() - ((d.getDay() + 3) % 7));
        return d;
      })();
      const expiry_dist = Math.max(0, (_lastThurs - _now) / 86400000) / 30;

      // hmm_regime as ordinal int (Bull=3, Chop=2, Bear=1, Panic=0)
      const _regimeInt = { 'Bull-Trend': 3, 'Bear-Trend': 1, 'Chop': 2, 'High-Vol-Panic': 0 };
      const _curRegime = mlEngine.getCurrentRegime();
      const hmm_regime_feat = _regimeInt[_curRegime] ?? 2;

      // ── Rolling CPR history (Sprint 1/2 CPR depth features) ──────────────────
      // histBars[n-1] = yesterday bar (= prevPeriod used to build today's cpr)
      // histBars[n-2] = day-before-yesterday → source for prev_cpr (yesterday's CPR)
      const _n = histBars.length;
      const _rollingCPR = idx => {
        if (idx < 0 || idx >= _n) return null;
        const b = histBars[idx];
        return calcCPR(b.high, b.low, b.close);
      };
      const prev_cpr = _rollingCPR(_n - 2);
      const _ydayClose = _n >= 1 ? histBars[_n - 1].close : last.close;

      // cpr_compress: today CPR width / 5-day avg CPR width
      let cpr_compress = 1;
      if (_n >= 7 && cpr.width > 0) {
        const p5w = [-2,-3,-4,-5,-6].map(i => { const c = _rollingCPR(_n + i); return c ? c.width : cpr.width; });
        const avg5w = p5w.reduce((a, b) => a + b, 0) / 5;
        cpr_compress = avg5w > 0 ? cpr.width / avg5w : 1;
      }

      // cpr_overlap_pct: overlap between today and yesterday CPR / today CPR width
      let cpr_overlap_pct = 0;
      if (prev_cpr && cpr.width > 0) {
        const ol = Math.max(0, Math.min(cpr.upper, prev_cpr.upper) - Math.max(cpr.lower, prev_cpr.lower));
        cpr_overlap_pct = ol / cpr.width;
      }

      const cpr_expansion_factor = (prev_cpr && prev_cpr.width > 0) ? cpr.width / prev_cpr.width : 1;
      const cpr_above_prev_cpr   = (prev_cpr && cpr.lower > prev_cpr.upper) ? 1 : 0;

      // prev_close_inside_cpr: yesterday close inside today's CPR band
      const prev_close_inside_cpr = (_ydayClose >= cpr.lower && _ydayClose <= cpr.upper) ? 1 : 0;

      // cpr_midpoint_trend: OLS slope of 5 prior CPR midpoints (normalized)
      let cpr_midpoint_trend = 0;
      if (_n >= 7) {
        const mids = [-6,-5,-4,-3,-2].map(i => { const c = _rollingCPR(_n + i); return c ? (c.bc + c.tc) / 2 : 0; });
        const ref = mids[0] || 1;
        const nm = mids.map(m => m / ref - 1);
        const nLen = nm.length, sx = nLen*(nLen-1)/2, sx2 = nLen*(nLen-1)*(2*nLen-1)/6;
        const sy = nm.reduce((a,b)=>a+b,0), sxy = nm.reduce((s,v,i)=>s+i*v,0);
        const den = nLen*sx2 - sx*sx;
        cpr_midpoint_trend = den !== 0 ? (nLen*sxy - sx*sy) / den : 0;
      }

      // consecutive_narrow_cprs: bars of consecutive prior CPRs narrower than today's
      let consecutive_narrow_cprs = 0;
      for (let i = _n - 2; i >= Math.max(0, _n - 11); i--) {
        const c = _rollingCPR(i);
        if (c && c.widthPct < cpr.widthPct) consecutive_narrow_cprs++;
        else break;
      }

      // cpr_width_percentile_252d: today CPR width percentile in last 252 days
      let cpr_width_percentile_252d = 0.5;
      if (_n >= 20) {
        const lookback = Math.min(252, _n - 1);
        const ws = [];
        for (let i = _n - 1 - lookback; i < _n - 1; i++) { const c = _rollingCPR(i); if (c) ws.push(c.width); }
        if (ws.length > 0) {
          ws.sort((a, b) => a - b);
          cpr_width_percentile_252d = ws.filter(w => w <= cpr.width).length / ws.length;
        }
      }

      // open_to_cpr_dist: (open - CPR midpoint) / CPR width
      const _cprMid = (cpr.bc + cpr.tc) / 2;
      const open_to_cpr_dist = cpr.width > 0 ? (last.open - _cprMid) / cpr.width : 0;
      const open_inside_cpr  = (last.open >= cpr.lower && last.open <= cpr.upper) ? 1 : 0;

      // atr_to_cpr_ratio: ATR14 / CPR width
      const _atr14 = (() => {
        if (hHighs.length < 2) return hHighs.length ? hHighs[0] - hLows[0] : 0;
        const len = Math.min(14, hHighs.length - 1);
        let s = 0;
        for (let i = hHighs.length - len; i < hHighs.length; i++) {
          const pc = hCloses[i - 1];
          s += Math.max(hHighs[i] - hLows[i], Math.abs(hHighs[i] - pc), Math.abs(hLows[i] - pc));
        }
        return s / len;
      })();
      const atr_to_cpr_ratio = cpr.width > 0 ? _atr14 / cpr.width : 1;

      // prev_cpr_respected: yesterday close inside yesterday's CPR
      const prev_cpr_respected = (prev_cpr && _ydayClose >= prev_cpr.lower && _ydayClose <= prev_cpr.upper) ? 1 : 0;

      // prev_bar_close_pos: yesterday close position within yesterday's CPR (0=below bc, 1=above tc)
      let prev_bar_close_pos = 0.5;
      if (prev_cpr && (prev_cpr.tc - prev_cpr.bc) > 0) {
        prev_bar_close_pos = Math.max(0, Math.min(1, (_ydayClose - prev_cpr.bc) / (prev_cpr.tc - prev_cpr.bc)));
      }

      // prev_day_ochoa_type: yesterday candle body classification (0=doji,1=bearish,2=neutral,3=bullish)
      let prev_day_ochoa_type = 2;
      if (_n >= 1) {
        const yb = histBars[_n - 1];
        const range = yb.high - yb.low;
        if (range > 0) {
          const bodyPct = Math.abs(yb.close - yb.open) / range;
          if (bodyPct < 0.1)        prev_day_ochoa_type = 0;
          else if (yb.close > yb.open) prev_day_ochoa_type = bodyPct > 0.6 ? 3 : 2;
          else                         prev_day_ochoa_type = bodyPct > 0.6 ? 1 : 2;
        }
      }

      // rsi_div: RSI divergence — price and RSI moving in opposite directions (-1/0/1)
      let rsi_div = 0;
      if (hCloses.length >= 6) {
        const priceSlope = hCloses[hCloses.length - 1] - hCloses[hCloses.length - 6];
        const rsi5ago    = calcRSI(hCloses.slice(0, hCloses.length - 5), 14);
        if      (priceSlope > 0 && xRsi14 < rsi5ago) rsi_div = -1;
        else if (priceSlope < 0 && xRsi14 > rsi5ago) rsi_div =  1;
      }

      // vol_accel_delta: change in vol_accel vs prior bar
      let vol_accel_delta = 0;
      if (hVols.length >= 21) {
        const vol3p  = (hVols[hVols.length-2] + hVols[hVols.length-3] + hVols[hVols.length-4]) / 3;
        const vol20p = hVols.slice(-21, -1).reduce((a, b) => a + b, 0) / 20;
        vol_accel_delta = vol_accel - (vol20p > 0 ? vol3p / vol20p : 1);
      }

      featureList = rulesToPredict.map(rid => {
        const direction  = getRuleDirection(rid, cpr, cam, last.close, prevClose, periodHigh, periodLow);
        const ruleNum    = parseInt(rid.replace('rule', ''));
        const vol_rank   = (hVols[hVols.length-1] || 0) / xVol20a;
        // Tier-1 interaction features (matches build_dataset.py computation)
        const conf_vol   = vixVal * vol_rank;
        const rsi_dir    = xRsi14 * direction;
        const hi52_dir   = dist_hi52 * direction;
        return {
          cpr_width_pct: cpr.widthPct,
          vwap_dist:     vwap > 0 ? (last.close - vwap) / vwap : 0,
          atr_pct_rank:  calcATRRank(hHighs, hLows, hCloses),
          vol_rank,
          n_rules_fired: qualityPassedRules.length || matchedRules.length,
          sg_vel:        xSgVel,
          ema200_dist:   ema200val > 0 ? (last.close - ema200val) / ema200val : 0,
          rsi14:         xRsi14,
          mom5:          xMom5,
          dow,
          rule_id:       ruleNum,
          direction,
          // Phase A: 52-week context + volume acceleration
          dist_hi52,
          dist_lo52,
          vol_accel,
          // Phase D: VIX (market_rs_*, sector_rs_*, deliv_pct, pcr not available real-time)
          india_vix:     vixVal,
          // Tier-1 interaction features
          conf_vol,
          rsi_dir,
          hi52_dir,
          // Sprint 3 + momentum features
          cpr_pos:        (cpr.tc - cpr.bc) > 0 ? (last.close - cpr.bc) / (cpr.tc - cpr.bc) : 0.5,
          dist_r1:        last.close > 0 ? (last.close - cam.r1) / last.close : 0,
          dist_s1:        last.close > 0 ? (last.close - cam.s1) / last.close : 0,
          mom3:           xMom3,
          mom10:          xMom10,
          mom20:          xMom20,
          gap_pct,
          atr_expansion,
          vol_trend_slope,
          days_since_52hi,
          expiry_dist,
          hmm_regime:     hmm_regime_feat,
          // Sprint 1 CPR depth features
          cpr_overlap_pct,
          open_to_cpr_dist,
          prev_cpr_respected,
          // Sprint 2A CPR depth features
          open_inside_cpr,
          consecutive_narrow_cprs,
          cpr_midpoint_trend,
          cpr_expansion_factor,
          // Sprint 2B CPR depth features
          cpr_above_prev_cpr,
          prev_close_inside_cpr,
          atr_to_cpr_ratio,
          cpr_width_percentile_252d,
          prev_day_ochoa_type,
          // Additional computable features
          cpr_compress,
          rsi_div,
          vol_accel_delta,
          prev_bar_close_pos,
        };
      });

      const preds = await xgbPredict(featureList);
      if (preds) {
        rulesToPredict.forEach((rid, i) => {
          predReturnByRule[rid] = +(preds[i] * 100).toFixed(3);
        });
      }
    }

    // ── ML Ensemble + Regime + Position Size ──────────────────────────────────
    let mlResult = null;
    const currentRegime = mlEngine.getCurrentRegime();
    const regimeScore   = { 'Bull-Trend': 1.0, 'Bear-Trend': 0.4, 'Chop': 0.2, 'High-Vol-Panic': 0.0 }[currentRegime] ?? 0.5;
    const regimeAllowed = mlEngine.isRegimeAllowed(currentRegime);

    if (rulesToPredict.length > 0 && featureList && featureList.length > 0) {
      const firstFeat = featureList[0];
      const [ensRes, posSize] = await Promise.all([
        mlEngine.getEnsembleScore(firstFeat).catch(() => null),
        mlEngine.getPositionSize(firstFeat, currentRegime).catch(() => 0.5),
      ]);
      const confInt = ensRes
        ? { lower: ensRes.confLower, upper: ensRes.confUpper }
        : mlEngine.getConfidenceInterval(Object.values(predReturnByRule)[0] || 0);
      mlResult = {
        regime:       currentRegime,
        regimeScore,
        regimeAllowed,
        stackScore:   ensRes ? ensRes.stackScore : null,
        xgbScore:     ensRes ? ensRes.xgbScore   : null,
        lgbmScore:    ensRes ? ensRes.lgbmScore   : null,
        confLower:    confInt.lower,
        confUpper:    confInt.upper,
        positionSize: posSize,
      };
    } else {
      mlResult = { regime: currentRegime, regimeScore, regimeAllowed, positionSize: 0.5 };
    }

    const p = v => +v.toFixed(2);
    const pPos = last.close > cpr.upper ? 'above' : last.close < cpr.lower ? 'below' : 'inside';

    return {
      symbol,
      price:     p(last.close),
      open:      p(last.open ?? currentBars[0].open),
      high:      p(periodHigh),
      low:       p(periodLow),
      prevClose: p(prevClose),
      change,
      cpr: {
        tc: p(cpr.upper), pivot: p(cpr.pivot), bc: p(cpr.lower),
        lower: p(cpr.lower), upper: p(cpr.upper), widthPct: +cpr.widthPct.toFixed(3)
      },
      camarilla: cam,
      vwap:          p(vwap),
      periodHigh:    p(periodHigh),
      periodLow:     p(periodLow),
      pricePosition: pPos,
      ruleResults,
      matchedRules,
      matchCount:    matchedRules.length,
      // Quality gate results
      ruleGates,
      qualityPassedRules,
      confluenceScore: +confluenceScore.toFixed(2),
      confluencePass,
      globalGates: {
        adx:     { value: +adxVal.toFixed(1),    pass: gateADX },
        atrPct:  { value: +atrPctVal.toFixed(2), pass: gateATRPct },
        volSurge:{ value: +volRatio.toFixed(2),  pass: gateVol },
        vix:     { value: +vixVal.toFixed(2),    pass: gateVIX },
        niftyEma:{ niftyClose: +niftyClose.toFixed(0), ema20: +niftyEma20.toFixed(0) },
        poc:     { value: poc ? +poc.toFixed(2) : null, distPct: +pocDistPct.toFixed(2), aligned: pocAligned }
      },
      // Optimal exit levels per quality-passed rule
      exitLevelsByRule,
      // XGB predictions
      predReturnByRule,
      bestPred: Object.values(predReturnByRule).length
        ? Math.max(...Object.values(predReturnByRule))
        : null,
      xgbActive: xgbAvailable,
      // ML ensemble result
      ml: mlResult,
    };
  } catch (err) {
    return { symbol, error: err.message };
  }
}

// ─── ROUTES ───────────────────────────────────────────────────────────────────

app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/rules', (_req, res) => {
  res.json(Object.entries(RULES).map(([id, r]) => ({
    id, name: r.name, desc: r.desc, color: r.color,
    sharpe: RULE_SHARPE[id] || 0
  })));
});

app.get('/api/symbols', (_req, res) => res.json(NSE_SYMBOLS));

app.get('/api/lists', (_req, res) => {
  const lists = LIST_CONFIG
    .filter(cfg => STOCK_LISTS[cfg.name])
    .map(cfg => ({ name: cfg.name, count: STOCK_LISTS[cfg.name].length }));
  res.json(lists);
});

app.get('/api/market-status', (_req, res) => {
  const { close: niftyClose, ema20: niftyEma20 } = latestNiftyAndEma();
  res.json({
    vix:    { value: latestVix(), safe: latestVix() < 18 || latestVix() === 0 },
    nifty:  { close: niftyClose, ema20: niftyEma20, aboveEma: niftyClose >= niftyEma20 }
  });
});

app.get('/api/screen/stream', async (req, res) => {
  res.writeHead(200, {
    'Content-Type':  'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection':    'keep-alive',
    'X-Accel-Buffering': 'no'
  });

  const tf       = req.query.tf || '1d';
  const mode     = req.query.mode || 'any';
  const narrow   = parseFloat(req.query.narrow) || 0.5;
  const rules    = (req.query.rules || Object.keys(RULES).join(',')).split(',').filter(r => RULES[r]);
  const listName = req.query.list;
  const symStr   = req.query.symbols;
  const qualityFilter  = req.query.quality !== 'false';   // default: show quality gates
  const disabledGates  = (req.query.disabledGates || '').split(',').filter(Boolean);

  let symbols;
  if (listName && STOCK_LISTS[listName]) {
    symbols = STOCK_LISTS[listName];
  } else if (symStr) {
    symbols = symStr.split(',').filter(Boolean);
  } else {
    symbols = STOCK_LISTS['Nifty 50'] || NSE_SYMBOLS;
  }

  const emit = obj => { if (!res.writableEnded) res.write(`data: ${JSON.stringify(obj)}\n\n`); };

  emit({ type: 'start', total: symbols.length, tf, rules, mode,
         marketStatus: {
           vix:   { value: latestVix(), safe: latestVix() < 18 || latestVix() === 0 },
           nifty: latestNiftyAndEma()
         }});

  let done = 0, matched = 0;
  const t0 = Date.now();
  const BATCH = 5;

  for (let i = 0; i < symbols.length; i += BATCH) {
    if (res.writableEnded) break;
    const batch   = symbols.slice(i, i + BATCH);
    const results = await Promise.all(
      batch.map(s => processSymbol(s, tf, rules, { narrowThreshold: narrow, disabledGates }))
    );

    for (const result of results) {
      done++;
      if (result.error) {
        emit({ type: 'err', done, total: symbols.length, sym: result.symbol });
        continue;
      }

      // Use quality-passed rules for match check if quality filter on
      const effectiveMatches = qualityFilter
        ? result.qualityPassedRules
        : result.matchedRules;

      const passes = mode === 'all'
        ? effectiveMatches.length === rules.length
        : effectiveMatches.length > 0;

      if (passes) {
        matched++;
        emit({ type: 'tick', done, total: symbols.length, matched, result });
      } else {
        emit({ type: 'skip', done, total: symbols.length, matched, sym: result.symbol });
      }
    }

    if (i + BATCH < symbols.length) await new Promise(r => setTimeout(r, 120));
  }

  emit({ type: 'done', total: symbols.length, matched, elapsed: Date.now() - t0 });
  res.end();
});

// ─── TRAINING API ─────────────────────────────────────────────────────────────

const { spawn } = require('child_process');

const PYTHON      = 'C:\\Users\\drkkr\\AppData\\Local\\Programs\\Python\\Python310\\python.exe';
const RUN_ALL     = path.join(__dirname, 'scripts', 'ml', 'run_all.py');
const LAST_RUN_F  = path.join(__dirname, 'auto_retrain_cpr.last_run');

let trainJob = { proc: null, status: 'idle', log: [], startedAt: null, exitCode: null };

app.get('/api/train/status', (_req, res) => {
  let lastRun = null;
  try { lastRun = fs.readFileSync(LAST_RUN_F, 'utf8').trim(); } catch {}
  res.json({ status: trainJob.status, startedAt: trainJob.startedAt, lastRun });
});

app.post('/api/train/start', (req, res) => {
  if (trainJob.proc) return res.status(409).json({ error: 'Training already running' });

  const skipUpload = req.query.skipUpload === '1';
  const args = [RUN_ALL, '--skip-dataset'];
  if (skipUpload) args.push('--skip-upload');

  trainJob = { proc: null, status: 'starting', log: [], startedAt: new Date().toISOString(), exitCode: null };

  const proc = spawn(PYTHON, args, { cwd: __dirname });
  trainJob.proc = proc;
  trainJob.status = 'running';

  const pushLine = chunk => {
    chunk.toString().split(/\r?\n/).forEach(line => {
      if (line !== undefined) trainJob.log.push(line);
    });
  };
  proc.stdout.on('data', pushLine);
  proc.stderr.on('data', pushLine);

  proc.on('close', code => {
    trainJob.status = code === 0 ? 'done' : 'failed';
    trainJob.proc   = null;
    trainJob.exitCode = code;
    if (code === 0) {
      try { fs.writeFileSync(LAST_RUN_F, new Date().toISOString()); } catch {}
    }
  });

  res.json({ started: true });
});

app.post('/api/train/cancel', (_req, res) => {
  if (trainJob.proc) {
    trainJob.proc.kill('SIGTERM');
    trainJob.status = 'cancelled';
    trainJob.proc   = null;
  }
  res.json({ cancelled: true });
});

app.get('/api/train/stream', (req, res) => {
  res.writeHead(200, {
    'Content-Type':  'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection':    'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const emit = obj => { if (!res.writableEnded) res.write(`data: ${JSON.stringify(obj)}\n\n`); };

  let sent = 0;
  // Flush backlog immediately
  trainJob.log.slice(0, sent = trainJob.log.length).forEach(line => emit({ line }));
  emit({ status: trainJob.status, startedAt: trainJob.startedAt });

  const iv = setInterval(() => {
    while (sent < trainJob.log.length) emit({ line: trainJob.log[sent++] });
    emit({ status: trainJob.status });
    if (trainJob.status === 'done' || trainJob.status === 'failed' || trainJob.status === 'cancelled') {
      clearInterval(iv);
      if (!res.writableEnded) res.end();
    }
  }, 400);

  req.on('close', () => clearInterval(iv));
});

// Live price refresh for currently displayed symbols
app.get('/api/live-prices', async (req, res) => {
  const symbols = (req.query.symbols || '').split(',').map(s => s.trim()).filter(Boolean);
  if (!symbols.length) return res.json({});
  const results = {};
  for (const sym of symbols) {
    try {
      const bar = await fetchNSEQuoteBar(sym);
      results[sym] = { price: +bar.close.toFixed(2), high: +bar.high.toFixed(2), low: +bar.low.toFixed(2) };
    } catch {
      results[sym] = null;
    }
  }
  res.json(results);
});

app.get('*', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.listen(PORT, () => {
  console.log(`\n  NSE Smart Screener → http://localhost:${PORT}\n`);
});
