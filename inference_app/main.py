"""
FastAPI Inference App for Expanded COCO Benchmark
===================================================
Clean web UI for:
  1. Entering a query word
  2. Selecting the correct WordNet synset (meaning group)
  3. Running CLIPSeg inference with alpha-blended embeddings

Usage:
  cd inference_app/
  pip install fastapi uvicorn python-multipart jinja2
  python main.py
  # Open http://localhost:8000
"""

import sys, os, io, json, base64, random
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from expanded_benchmark.helpers import (
    COCO_80, load_coco, download_image, get_gt_mask,
    to_wn_form, to_display_form,
)
from expanded_benchmark.inference import (
    fetch_synset_groups,
    build_word_set_for_selected_synset,
    run_inference_pipeline,
)

app = FastAPI(title="Expanded COCO Benchmark — Inference")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── In-memory session store (simple dict, not for production) ──
sessions = {}

# ═══════════════════════════════════════════════
# HTML template (inline for simplicity)
# ═══════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Expanded COCO Benchmark — Inference</title>
    <style>
        :root {
            --bg: #0d1117; --surface: #161b22; --border: #30363d;
            --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
            --success: #3fb950; --warning: #d2991d; --danger: #f85149;
            --radius: 8px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg); color: var(--text);
            max-width: 960px; margin: 0 auto; padding: 2rem 1rem;
            line-height: 1.6;
        }
        h1 { font-size: 1.5rem; margin-bottom: .25rem; }
        h2 { font-size: 1.15rem; margin: 1.5rem 0 .75rem; color: var(--accent); }
        h3 { font-size: 1rem; margin: 1rem 0 .5rem; }
        .subtitle { color: var(--muted); font-size: .875rem; margin-bottom: 2rem; }

        .step {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 1.25rem; margin-bottom: 1.25rem;
        }
        .step-num {
            display: inline-block; background: var(--accent); color: #000;
            border-radius: 50%; width: 1.5rem; height: 1.5rem; line-height: 1.5rem;
            text-align: center; font-weight: 700; font-size: .8rem; margin-right: .5rem;
        }

        label { display: block; font-size: .875rem; color: var(--muted); margin-bottom: .25rem; }
        input[type="text"], input[type="number"], input[type="range"] {
            width: 100%; padding: .5rem .75rem; background: var(--bg);
            border: 1px solid var(--border); border-radius: var(--radius);
            color: var(--text); font-size: .9rem; margin-bottom: .75rem;
        }
        input:focus { outline: none; border-color: var(--accent); }
        button, .btn {
            display: inline-block; padding: .5rem 1.25rem;
            background: var(--accent); color: #000; border: none;
            border-radius: var(--radius); cursor: pointer; font-weight: 600;
            font-size: .875rem; margin-right: .5rem; margin-bottom: .5rem;
        }
        button:hover { opacity: .85; }
        button:disabled { opacity: .4; cursor: not-allowed; }

        .synset-card {
            border: 1px solid var(--border); border-radius: var(--radius);
            padding: .75rem 1rem; margin-bottom: .5rem; cursor: pointer;
            transition: border-color .15s;
        }
        .synset-card:hover { border-color: var(--accent); }
        .synset-card.selected { border-color: var(--success); background: rgba(63,185,80,.08); }
        .synset-card .name { font-weight: 600; font-size: .9rem; color: var(--accent); }
        .synset-card .defn { font-size: .8rem; color: var(--muted); margin: .25rem 0; }
        .synset-card .lemmas { font-size: .75rem; color: var(--muted); }

        .word-set-box {
            background: var(--bg); border: 1px solid var(--border);
            border-radius: var(--radius); padding: .75rem 1rem; margin-bottom: .5rem;
        }
        .word-set-box .label { font-weight: 600; font-size: .8rem; color: var(--accent); }
        .word-set-box .words { font-size: .8rem; color: var(--text); word-break: break-all; }

        .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .result-card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 1rem;
        }
        .result-card .title { font-weight: 600; font-size: .85rem; color: var(--accent); margin-bottom: .5rem; }
        .metric { display: flex; justify-content: space-between; font-size: .85rem; padding: .2rem 0; border-bottom: 1px solid var(--border); }
        .metric .val { font-weight: 600; }
        .metric .pos { color: var(--success); }
        .metric .neg { color: var(--danger); }

        .hidden { display: none; }
        .spinner {
            display: inline-block; width: 1rem; height: 1rem; border: 2px solid transparent;
            border-top-color: currentColor; border-radius: 50%; animation: spin .6s linear infinite;
            vertical-align: middle; margin-right: .5rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        #status-msg { font-size: .8rem; color: var(--warning); margin: .5rem 0; min-height: 1.2rem; }

        table { width: 100%; border-collapse: collapse; font-size: .8rem; }
        th, td { padding: .5rem; text-align: left; border-bottom: 1px solid var(--border); }
        th { color: var(--muted); font-weight: 600; }

        .range-group { display: flex; align-items: center; gap: .75rem; }
        .range-group input[type="range"] { flex: 1; margin-bottom: 0; }
        .range-group .range-val { font-weight: 600; min-width: 2.5rem; text-align: center; color: var(--accent); }
    </style>
</head>
<body>

<h1>🔬 Expanded COCO Benchmark — Inference</h1>
<p class="subtitle">WordNet-synset-disambiguated CLIPSeg segmentation with alpha blending</p>

<!-- Step 1: Word input -->
<div class="step" id="step1">
    <h2><span class="step-num">1</span> Enter a Query Word</h2>
    <label for="query-word">Word to segment (any noun — e.g., "bank", "couch", "mouse")</label>
    <div style="display:flex; gap:.5rem;">
        <input type="text" id="query-word" placeholder="e.g., bank, couch, mouse, bat..."
               style="flex:1;" onkeydown="if(event.key==='Enter') lookupSynsets()">
        <button onclick="lookupSynsets()">Look Up Synsets</button>
    </div>
    <div id="status-msg"></div>
    <div id="synset-results"></div>
</div>

<!-- Step 2: Selected synset & image -->
<div class="step hidden" id="step2">
    <h2><span class="step-num">2</span> Configure & Run Inference</h2>

    <div id="selected-synset-info" style="margin-bottom:1rem;"></div>

    <h3>Word Sets</h3>
    <div id="word-sets-display"></div>

    <h3>Image Source</h3>
    <div style="display:flex; gap:1rem; flex-wrap:wrap;">
        <div style="flex:1; min-width:200px;">
            <label for="coco-category">COCO Category (for GT mask)</label>
            <select id="coco-category" style="width:100%; padding:.5rem; background:var(--bg);
                    border:1px solid var(--border); border-radius:var(--radius); color:var(--text);">
                <option value="">-- Auto-detect --</option>
            </select>
        </div>
        <div style="flex:1; min-width:200px;">
            <label for="coco-img-id">COCO Image ID (optional)</label>
            <input type="text" id="coco-img-id" placeholder="Leave blank for random">
        </div>
        <div style="flex:0 0 auto;">
            <label>&nbsp;</label>
            <label style="display:flex; align-items:center; gap:.5rem; cursor:pointer;">
                <input type="file" id="upload-image" accept="image/*" style="display:none;"
                       onchange="handleImageUpload(event)">
                <button type="button" onclick="document.getElementById('upload-image').click()">📁 Upload Image</button>
            </label>
        </div>
    </div>
    <div id="upload-preview" style="margin:.5rem 0;"></div>

    <h3>Blending Parameters</h3>
    <label>Alpha (α): blending weight — higher = more influence from centroid</label>
    <div class="range-group">
        <span style="font-size:.8rem;color:var(--muted);">0.0 (raw query)</span>
        <input type="range" id="alpha" min="0" max="10" step="1" value="7"
               oninput="document.getElementById('alpha-val').textContent=(this.value/10).toFixed(1)">
        <span class="range-val" id="alpha-val">0.7</span>
        <span style="font-size:.8rem;color:var(--muted);">1.0 (centroid)</span>
    </div>

    <label for="n-neighbors" style="margin-top:.75rem;">Top-K synonym neighbors</label>
    <input type="number" id="n-neighbors" value="5" min="1" max="20" style="width:6rem;">

    <button onclick="runInference()" style="margin-top:1rem; font-size:1rem; padding:.6rem 2rem;">
        🚀 Run Inference
    </button>
</div>

<!-- Step 3: Results -->
<div class="step hidden" id="step3">
    <h2><span class="step-num">3</span> Results</h2>
    <div id="results-content"></div>
</div>

<script>
let selectedSynset = null;
let uploadedImageB64 = null;

// Populate COCO categories dropdown
const COCO_80 = {{ COCO_80 | tojson }};
window.addEventListener('DOMContentLoaded', () => {
    const sel = document.getElementById('coco-category');
    COCO_80.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c;
        sel.appendChild(opt);
    });
});

function setStatus(msg, isError=false) {
    const el = document.getElementById('status-msg');
    el.textContent = msg;
    el.style.color = isError ? 'var(--danger)' : 'var(--warning)';
}

async function apiPost(url, data) {
    const resp = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data),
    });
    return resp.json();
}

async function lookupSynsets() {
    const word = document.getElementById('query-word').value.trim();
    if (!word) { setStatus('Please enter a word', true); return; }
    setStatus('Looking up WordNet synsets...');
    document.getElementById('synset-results').innerHTML = '';

    const data = await apiPost('/api/synsets', {word});
    if (data.error) { setStatus(data.error, true); return; }

    setStatus(`Found ${data.num_synsets} synset group(s) for "${data.query_word}"`);

    let html = '';
    data.synsets.forEach((s, i) => {
        html += `
        <div class="synset-card" onclick="selectSynset('${s.synset_name}', this)" id="card-${i}">
            <div class="name">${s.synset_name} (${s.pos}) — ${s.lemma_count} lemmas</div>
            <div class="defn">${s.definition}</div>
            <div class="lemmas">${s.lemma_names_display.slice(0,8).join(', ')}${s.total_lemmas > 8 ? ', ...' : ''}</div>
        </div>`;
    });

    if (data.synsets.length === 0) {
        html = '<p style="color:var(--warning);">No synsets found. Try a different word.</p>';
    }

    document.getElementById('synset-results').innerHTML = html;
}

function selectSynset(name, cardEl) {
    selectedSynset = name;
    document.querySelectorAll('.synset-card').forEach(c => c.classList.remove('selected'));
    cardEl.classList.add('selected');

    // Load word sets
    loadWordSets(name);
}

async function loadWordSets(synsetName) {
    setStatus('Loading word sets...');
    const data = await apiPost('/api/build-word-set', {synset_name: synsetName});
    if (data.error) { setStatus(data.error, true); return; }

    let html = `<p style="font-size:.85rem; color:var(--muted);">
        <strong>${data.synset_name}</strong>: ${data.definition}</p>`;

    for (const [level, words] of [
        ['W_S (Synonyms)', data.W_S_display],
        ['W_S_Hp (+Hyponyms)', data.W_S_Hp_display],
        ['W_S_Hp_He (+Hypernyms)', data.W_S_Hp_He_display],
    ]) {
        html += `<div class="word-set-box">
            <div class="label">${level} — ${words.length} words</div>
            <div class="words">${words.join(', ')}</div>
        </div>`;
    }

    if (data.babelnet_supplemented) {
        html += '<p style="font-size:.75rem; color:var(--warning);">⚠️ Supplemented with BabelNet</p>';
    }

    document.getElementById('word-sets-display').innerHTML = html;
    document.getElementById('selected-synset-info').innerHTML =
        `<strong>Selected:</strong> <span style="color:var(--accent);">${data.synset_name}</span>`;

    document.getElementById('step2').classList.remove('hidden');
    document.getElementById('step3').classList.add('hidden');
    setStatus('Ready — configure image and parameters, then run inference.');
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImageB64 = e.target.result.split(',')[1];
        document.getElementById('upload-preview').innerHTML =
            `<img src="${e.target.result}" style="max-width:200px; max-height:150px; border-radius:var(--radius); border:1px solid var(--border);">`;
    };
    reader.readAsDataURL(file);
}

async function runInference() {
    if (!selectedSynset) { setStatus('Select a synset first', true); return; }
    setStatus('Running inference...');

    const word = document.getElementById('query-word').value.trim();
    const alpha = parseInt(document.getElementById('alpha').value) / 10;
    const nNeighbors = parseInt(document.getElementById('n-neighbors').value);
    const cocoCat = document.getElementById('coco-category').value || null;
    const cocoImgId = document.getElementById('coco-img-id').value || null;
    const threshold = 0.5;

    const payload = {
        query_word: word,
        synset_name: selectedSynset,
        alpha, n_neighbors: nNeighbors, threshold,
        coco_category: cocoCat,
        img_id: cocoImgId ? parseInt(cocoImgId) : null,
        uploaded_image_b64: uploadedImageB64,
    };

    setStatus('Running CLIPSeg inference (this may take a moment)...');
    const data = await apiPost('/api/inference', payload);
    if (data.error) { setStatus(data.error, true); return; }

    displayResults(data);
    setStatus('');
}

function displayResults(data) {
    document.getElementById('step3').classList.remove('hidden');
    document.getElementById('step3').scrollIntoView({behavior: 'smooth'});

    let html = '<div class="result-grid">';

    // Left: Metrics
    html += '<div class="result-card"><div class="title">📊 Metrics</div>';
    html += `<div class="metric"><span>Query word</span><span class="val">${data.query_word}</span></div>`;
    html += `<div class="metric"><span>Synset</span><span class="val">${data.synset_name}</span></div>`;
    html += `<div class="metric"><span>Technique</span><span class="val">${data.technique}</span></div>`;
    html += `<div class="metric"><span>Alpha</span><span class="val">${data.alpha}</span></div>`;

    if (data.iou_raw_query !== null) {
        const delta = data.iou_delta;
        const cls = delta >= 0 ? 'pos' : 'neg';
        html += `<div class="metric"><span>IoU (raw query)</span><span class="val">${data.iou_raw_query}</span></div>`;
        html += `<div class="metric"><span>IoU (blended)</span><span class="val">${data.iou_blended}</span></div>`;
        html += `<div class="metric"><span>ΔIoU</span><span class="val ${cls}">${delta >= 0 ? '+' : ''}${delta}</span></div>`;
    }

    html += `<div class="metric"><span>Query norm</span><span class="val">${data.query_norm}</span></div>`;
    html += `<div class="metric"><span>Blended norm</span><span class="val">${data.blended_norm}</span></div>`;
    html += `<div class="metric"><span>Query-centroid cos</span><span class="val">${data.query_centroid_cosine}</span></div>`;
    html += `<div class="metric"><span>Image ID</span><span class="val">${data.img_id || 'N/A'}</span></div>`;
    html += `<div class="metric"><span>COCO category</span><span class="val">${data.coco_category || 'N/A'}</span></div>`;
    html += `<div class="metric"><span>Image size</span><span class="val">${data.image_size ? data.image_size.join('×') : 'N/A'}</span></div>`;
    html += '</div>';

    // Right: Top-K neighbors
    html += '<div class="result-card"><div class="title">🔝 Top-K Synonym Neighbors</div>';
    html += '<table><tr><th>#</th><th>Word</th><th>CLIP Cos Sim</th></tr>';
    data.top_k_neighbors.forEach((n, i) => {
        html += `<tr><td>${i+1}</td><td style="color:var(--accent);">${toDisplay(n.word)}</td><td>${n.similarity}</td></tr>`;
    });
    html += '</table></div>';

    // Word sets
    html += '<div class="result-card"><div class="title">📝 Word Sets Used</div>';
    html += '<table><tr><th>Level</th><th>Count</th></tr>';
    html += `<tr><td>W_S (Synonyms)</td><td>${data.word_sets.W_S.length}</td></tr>`;
    html += `<tr><td>W_S_Hp (+Hyponyms)</td><td>${data.word_sets.W_S_Hp.length}</td></tr>`;
    html += `<tr><td>W_S_Hp_He (+Hypernyms)</td><td>${data.word_sets.W_S_Hp_He.length}</td></tr>`;
    html += '</table></div>';

    // Mask preview info
    html += '<div class="result-card"><div class="title">🖼️ Mask Info</div>';
    html += `<p style="font-size:.8rem;">Prediction mask shape: ${data.pred_mask_shape || 'computed'}</p>`;
    html += `<p style="font-size:.8rem; color:var(--muted);">Mask arrays returned in response (use programmatically for visualization)</p>`;
    html += '</div>';

    html += '</div>';
    document.getElementById('results-content').innerHTML = html;
}

function toDisplay(w) { return w.replace(/_/g, ' '); }
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    """Main page."""
    return HTML_TEMPLATE.replace("{{ COCO_80 | tojson }}", json.dumps(COCO_80))


@app.post("/api/synsets")
async def api_synsets(request: Request):
    """Look up WordNet synset groups for a word."""
    data = await request.json()
    word = data.get("word", "").strip()
    if not word:
        return JSONResponse({"error": "No word provided"}, status_code=400)
    result = fetch_synset_groups(word)
    return JSONResponse(result)


@app.post("/api/build-word-set")
async def api_build_word_set(request: Request):
    """Build word sets for a selected synset."""
    data = await request.json()
    synset_name = data.get("synset_name", "").strip()
    if not synset_name:
        return JSONResponse({"error": "No synset name provided"}, status_code=400)
    result = build_word_set_for_selected_synset(synset_name)
    return JSONResponse(result)


@app.post("/api/inference")
async def api_inference(request: Request):
    """Run the full inference pipeline."""
    data = await request.json()
    query_word = data.get("query_word", "").strip()
    synset_name = data.get("synset_name", "").strip()
    alpha = float(data.get("alpha", 0.7))
    n_neighbors = int(data.get("n_neighbors", 5))
    threshold = float(data.get("threshold", 0.5))
    coco_category = data.get("coco_category") or None
    img_id = data.get("img_id") or None
    uploaded_image_b64 = data.get("uploaded_image_b64") or None

    if not query_word or not synset_name:
        return JSONResponse({"error": "Missing query_word or synset_name"}, status_code=400)

    # Load image
    image = None
    gt_mask = None
    image_source = "unknown"

    if uploaded_image_b64:
        try:
            img_bytes = base64.b64decode(uploaded_image_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_source = "upload"
        except Exception as e:
            return JSONResponse({"error": f"Failed to decode uploaded image: {e}"}, status_code=400)
    else:
        # Use COCO image
        try:
            coco = load_coco()
        except Exception:
            return JSONResponse({"error": "COCO annotations not found and no image uploaded"}, status_code=400)

        # Determine category
        cat_id = None
        if coco_category:
            cat_name_to_id = {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}
            cat_id = cat_name_to_id.get(coco_category)

        # Pick image
        if img_id is not None:
            img_id = int(img_id)
        else:
            if cat_id:
                ann_ids = coco.getAnnIds(catIds=[cat_id])
                all_img_ids = list(set(
                    coco.loadAnns(ann_ids)[i]["image_id"]
                    for i in range(min(len(ann_ids), 200))
                ))
                if all_img_ids:
                    img_id = random.choice(all_img_ids)
                else:
                    img_id = coco.getImgIds()[0]
            else:
                img_id = coco.getImgIds()[0]

        image = download_image(coco, img_id)
        if image is None:
            return JSONResponse({"error": f"Failed to download COCO image {img_id}"}, status_code=500)

        if cat_id:
            gt_mask = get_gt_mask(coco, img_id, cat_id)

        image_source = f"coco/{img_id}"

    # Run inference
    result = run_inference_pipeline(
        query_word=query_word,
        synset_name=synset_name,
        image=image,
        alpha=alpha,
        n_neighbors=n_neighbors,
        threshold=threshold,
        gt_mask=gt_mask,
    )

    # Remove numpy arrays from JSON response (they're large)
    pred_mask_shape = result.get("pred_mask", np.array([])).shape if result.get("pred_mask") is not None else None
    result.pop("pred_mask", None)
    result.pop("raw_pred_mask", None)
    result["pred_mask_shape"] = pred_mask_shape
    result["image_source"] = image_source

    return JSONResponse(result)


# ═══════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting Expanded COCO Benchmark — Inference Server")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
