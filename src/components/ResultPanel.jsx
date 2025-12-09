import React from "react";

export default function ResultPanel({ result, model=null }){
  if(!result) {
    return <div className="bg-slate-50 p-3 rounded text-slate-600">No prediction yet</div>;
  }

  // Normalize payload: API sometimes wraps the actual results under `result`.
  const payload = result?.result ?? result;
  const requestedModel = (model || result?.model_requested || null);

  // Determine which key to show (be permissive about backend naming)
  function getModelResultForKey(res, m){
    if(!res || !m) return null;
    const lower = m.toLowerCase();
    const candidates = [];
    if(lower === 'hybrid') candidates.push('hybrid_cnn', 'hybrid');
    if(lower === 'cnn') candidates.push('cnn');
    if(lower === 'sklearn' || lower === 'sklearn_model') candidates.push('sklearn', 'sklearn_model');
    // fallback: try exact key names
    for(const k of candidates){ if(res && Object.prototype.hasOwnProperty.call(res, k) && res[k]) return res[k]; }
    // as last resort, return a key even if null (so caller can show explicit 'null')
    for(const k of candidates){ if(res && Object.prototype.hasOwnProperty.call(res, k)) return res[k]; }
    return null;
  }

  const single = requestedModel ? getModelResultForKey(payload, requestedModel) : null;

  function renderSingle(r){
    if(!r) return <div className="text-sm text-slate-600">No result for selected model.</div>;
    if(r.error) return <div className="text-sm text-red-600">Error: {r.error}</div>;
    const probs = r.probs || null;
    const label = r.label || 'Unknown';
    let confidence = null;
    if(Array.isArray(probs)){
      const max = Math.max(...probs.map(p=>p||0));
      confidence = (max*100).toFixed(2) + '%';
    }
    return (
      <div>
        <div className="text-sm text-slate-600">Label</div>
        <div className="font-semibold text-lg">{label}</div>
        {confidence && <div className="text-xs text-slate-500 mt-1">Confidence: {confidence}</div>}
      </div>
    );
  }

  return (
    <div className="result-panel p-4 rounded">
      <h4 className="font-semibold mb-2">Prediction</h4>
      {model ? renderSingle(single) : (
        <div>
          <div className="text-sm text-slate-600 mb-2">Ensemble / All models</div>
          <pre aria-label="prediction-result" className="text-xs overflow-auto max-h-64 bg-slate-50 p-2 rounded">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
