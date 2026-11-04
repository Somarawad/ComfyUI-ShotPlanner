/*
 * Open every workflow in examples/workflows/ in a running ComfyUI with Playwright and check it:
 *   - it loads without page errors, console errors or an error dialog;
 *   - which node types are missing (third-party packs not installed) is reported, nothing else;
 *   - no two nodes overlap and every node sits inside one group;
 *   - the canvas converts back to the API prompt it was made from (app.graphToPrompt), node by
 *     node and input by input, so the UI file queues exactly what the API prompt describes;
 *   - a screenshot of the whole canvas is saved to docs/img/<workflow>.png.
 *
 * Setup: npm i playwright && npx playwright install chromium   (or use an existing install)
 * Run (ComfyUI running with this pack, e.g. python main.py --cpu):
 *   node tools/check_workflows.js --url http://127.0.0.1:8188
 * Options: --dir <folder> (default examples/workflows), --only 03_novara_stills_spot.json, --measure sizes.json (write the node sizes the
 * frontend draws, for python tools/api_to_workflow.py --all --sizes sizes.json), --no-screenshots.
 * Exit code 0 = every workflow passed.
 */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i < 0 ? def : (process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[i + 1] : true);
}

const ROOT = path.resolve(__dirname, '..');
const WF_DIR = path.resolve(arg('--dir', path.join(ROOT, 'examples', 'workflows')));
const API_DIR = path.join(ROOT, 'examples', 'api');
const IMG_DIR = path.join(ROOT, 'docs', 'img');


const url = arg('--url', 'http://127.0.0.1:8188');
const only = arg('--only', null);
const measure = arg('--measure', null);
const shots = !process.argv.includes('--no-screenshots');

function loadRename() {
  // the same portable-name map the converter applied (tools/workflow_recipes.py RENAME)
  const { execFileSync } = require('child_process');
  const py = process.env.PYTHON || 'python3';
  const out = execFileSync(py, ['-c',
    'import json,sys; sys.path.insert(0, "tools"); from workflow_recipes import RECIPES; '
    + 'print(json.dumps({r["out"]: [r["api"], r.get("rename", {})] for r in RECIPES}))'], { cwd: ROOT });
  return JSON.parse(out.toString());
}

function same(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-6;
  return JSON.stringify(a) === JSON.stringify(b);
}

function declaredDefault(info, name) {
  // default of an input as /object_info declares it (V3 dynamic inputs like "format.codec" use the
  // first option of the combo they belong to)
  for (const sec of ['required', 'optional']) {
    const spec = (info.input && info.input[sec] || {})[name];
    if (!spec) continue;
    const opts = spec[1] || {};
    if ('default' in opts) return { ok: true, value: opts.default };
    if (Array.isArray(spec[0]) && spec[0].length) return { ok: true, value: spec[0][0] };
    if (spec[0] === 'COMBO' && (opts.options || []).length) return { ok: true, value: opts.options[0] };
  }
  return { ok: false };
}

function compare(api, rename, out, objectInfo) {
  // UI ids are 1..n in API order (see api_to_workflow.py); links become [id, slot]
  const keys = Object.keys(api);
  const idOf = Object.fromEntries(keys.map((k, i) => [k, String(i + 1)]));
  const problems = [];
  const missingTypes = new Set();
  const notes = [];
  for (const k of keys) {
    const want = api[k];
    const got = out[idOf[k]];
    const info = objectInfo[want.class_type];
    if (!info) { missingTypes.add(want.class_type); continue; }  // pack not installed: nothing to compare
    if (!got) { problems.push(`${k}: ${want.class_type} is not in the prompt built from the canvas`); continue; }
    if (got.class_type !== want.class_type) problems.push(`${k}: class ${got.class_type} != ${want.class_type}`);
    for (const [name, v] of Object.entries(want.inputs)) {
      let w = Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' ? [idOf[v[0]], v[1]] : v;
      if (typeof w === 'string' && rename[w] !== undefined) w = rename[w];
      if (!(name in got.inputs)) { problems.push(`${k}.${name}: missing on the canvas`); continue; }
      if (!same(got.inputs[name], w)) problems.push(`${k}.${name}: ${JSON.stringify(got.inputs[name]).slice(0, 80)} != ${JSON.stringify(w).slice(0, 80)}`);
    }
    for (const name of Object.keys(got.inputs)) {
      if (name in want.inputs) continue;
      // the API prompt left an optional input out (= its default); the canvas sends the widget value
      const d = declaredDefault(info, name);
      if (d.ok && same(got.inputs[name], d.value)) { notes.push(`${k}.${name}`); continue; }
      const sub = name.split('.');
      if (sub.length > 1) { notes.push(`${k}.${name}`); continue; }  // V3 dynamic sub-input
      problems.push(`${k}.${name}: canvas sends ${JSON.stringify(got.inputs[name]).slice(0, 60)}, which is not the default`);
    }
  }
  return { problems, missingTypes: [...missingTypes], defaultsAdded: notes.length };
}

(async () => {
  const recipes = loadRename();
  const objectInfo = await (await fetch(`${url}/object_info`)).json();
  // Clean canvas for the checks and screenshots: no first-run dialog, no missing-model / node
  // pop-ups (missing types are found by the round trip instead), full detail when zoomed out.
  // The previous values are put back at the end.
  const SETTINGS = {
    'Comfy.TutorialCompleted': true, 'Comfy.ErrorSystem.ShowMissingModels': false,
    'Comfy.Workflow.ShowMissingNodesWarning': false, 'Comfy.Workflow.ShowMissingMediaWarning': false,
    'Comfy.Manager.ShowMissingPacks': false, 'Comfy.Minimap.Visible': false,
    'LiteGraph.Canvas.LowQualityRenderingZoomThreshold': 0.1,
  };
  const previous = {};
  for (const [k, v] of Object.entries(SETTINGS)) {
    try {
      previous[k] = await (await fetch(`${url}/settings/${k}`)).json();
      await fetch(`${url}/settings/${k}`, { method: 'POST', body: JSON.stringify(v) });
    } catch (e) { /* older ComfyUI: setting not there */ }
  }
  const files = fs.readdirSync(WF_DIR).filter(f => f.endsWith('.json') && (!only || f === only)).sort();
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 2400, height: 1350 }, deviceScaleFactor: 1.5 });
  let log = [];
  page.on('console', m => { if (m.type() === 'error') log.push('console: ' + m.text()); });
  page.on('pageerror', e => log.push('pageerror: ' + e.message));
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 60000 });
  await page.keyboard.press('Escape');
  if (shots) await page.evaluate(() => window.app.extensionManager.command.execute('Workspace.ToggleFocusMode'));
  await page.waitForTimeout(1500);
  const sizes = {};
  let failed = 0;

  for (const f of files) {
    const wf = JSON.parse(fs.readFileSync(path.join(WF_DIR, f), 'utf8'));
    log = [];
    await page.evaluate(async (data) => {
      await window.app.loadGraphData(data, true, true, null);
    }, wf);
    await page.waitForTimeout(2500);
    const dialogs = await page.evaluate(() => [...document.querySelectorAll('.p-dialog')]
      .map(d => d.innerText.replace(/\s+/g, ' ').slice(0, 400)));
    const state = await page.evaluate(() => {
      const g = window.app.graph;
      const nodes = g._nodes.map(n => ({ id: n.id, type: n.type, pos: [...n.pos], size: [...n.size],
        title_h: window.LiteGraph ? window.LiteGraph.NODE_TITLE_HEIGHT : 30, has_errors: !!n.has_errors }));
      const groups = g._groups.map(gr => ({ title: gr.title, b: [...gr._bounding] }));
      return { nodes, groups };
    });
    let out = {};
    try {
      out = await page.evaluate(async () => (await window.app.graphToPrompt()).output);
    } catch (e) { log.push('graphToPrompt: ' + e.message); }

    const problems = [];
    // 1. load errors (dialogs other than the missing-nodes report count as errors)
    const ignorable = (t) => /Failed to load resource|ERR_TUNNEL|net::ERR_/.test(t);
    for (const l of log) if (!ignorable(l)) problems.push(l);
    const missingDialog = dialogs.filter(d => /missing|not found|not installed/i.test(d));
    for (const d of dialogs) if (!missingDialog.includes(d)) problems.push('dialog: ' + d);
    // 2. layout: overlaps and group membership
    const rect = n => [n.pos[0], n.pos[1] - n.title_h, n.pos[0] + n.size[0], n.pos[1] + n.size[1]];
    const N = state.nodes;
    for (let i = 0; i < N.length; i++) for (let j = i + 1; j < N.length; j++) {
      const a = rect(N[i]), b = rect(N[j]);
      if (a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]) problems.push(`overlap: #${N[i].id} ${N[i].type} / #${N[j].id} ${N[j].type}`);
    }
    for (const n of (state.groups.length ? N : [])) {  // workflows without groups: nothing to check
      const r = rect(n);
      const inside = state.groups.filter(g => r[0] >= g.b[0] && r[1] >= g.b[1] && r[2] <= g.b[0] + g.b[2] && r[3] <= g.b[1] + g.b[3]);
      if (inside.length !== 1) problems.push(`#${n.id} ${n.type} is in ${inside.length} groups`);
    }
    // 3. round trip against the source API prompt
    let rt = { problems: [], missingTypes: [], defaultsAdded: 0 };
    if (recipes[f]) {
      const [apiFile, rename] = recipes[f];
      const api = JSON.parse(fs.readFileSync(path.join(API_DIR, apiFile), 'utf8'));
      rt = compare(api, rename, out, objectInfo);
      problems.push(...rt.problems.map(p => 'round trip: ' + p));
    }

    if (measure) sizes[f] = Object.fromEntries(N.map(n => [String(n.id), n.size.map(Math.ceil)]));
    if (shots && !problems.length) {
      await page.evaluate(() => {
        for (const d of document.querySelectorAll('.p-dialog-close-button, .p-dialog-header-close, .p-toast-close-button')) d.click();
      });
      await page.keyboard.press('Escape');
      await page.evaluate(() => window.app.extensionManager.command.execute('Comfy.Canvas.FitView'));
      await page.waitForTimeout(1200);
      // crop to the groups (no empty margins, no canvas controls or render stats)
      const clip = await page.evaluate(() => {
        for (const t of document.querySelectorAll('.p-toast')) t.style.display = 'none';
        const c = window.app.canvas;
        c.render_info = false;
        c.setDirty(true, true);
        const gs = window.app.graph._groups.map(g => g._bounding);
        const x0 = Math.min(...gs.map(b => b[0])), y0 = Math.min(...gs.map(b => b[1]));
        const x1 = Math.max(...gs.map(b => b[0] + b[2])), y1 = Math.max(...gs.map(b => b[1] + b[3]));
        const r = c.canvas.getBoundingClientRect(), s = c.ds.scale, o = c.ds.offset, m = 16;
        const sx = r.left + (x0 + o[0]) * s - m, sy = r.top + (y0 + o[1]) * s - m;
        return { x: Math.max(0, sx), y: Math.max(0, sy),
                 width: Math.min(r.width, (x1 - x0) * s + 2 * m), height: Math.min(r.height, (y1 - y0) * s + 2 * m) };
      });
      await page.waitForTimeout(400);
      fs.mkdirSync(IMG_DIR, { recursive: true });
      await page.screenshot({ path: path.join(IMG_DIR, f.replace(/\.json$/, '.png')), clip });
    }
    const status = problems.length ? 'FAIL' : 'ok  ';
    if (problems.length) failed++;
    console.log(`${status} ${f}: ${N.length} nodes, ${state.groups.length} groups`
      + (rt.missingTypes.length ? `, not installed here: ${rt.missingTypes.join(', ')}` : '')
      + (recipes[f] ? (rt.problems.length ? ', round trip DIFFERS from the API prompt'
        : `, round trip = API prompt (${rt.defaultsAdded} omitted optional inputs sent as their defaults)`) : ''));
    for (const p of problems.slice(0, 40)) console.log('      ' + p);
  }
  if (measure) fs.writeFileSync(measure, JSON.stringify(sizes, null, 1));
  await browser.close();
  for (const [k, v] of Object.entries(previous)) {
    await fetch(`${url}/settings/${k}`, { method: 'POST', body: JSON.stringify(v) }).catch(() => {});
  }
  console.log(failed ? `\nFAILED: ${failed} workflow(s)` : '\nPASSED');
  process.exit(failed ? 1 : 0);
})();
