import React, { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react";

const MAX_BYTES = 100 * 1024 * 1024; // 100MB

function UploadPanel({ onResult, apiBase = "http://localhost:8000", model = null, onLoadingChange = null, onPreview = null }, ref){
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const dropRef = useRef(null);
  const fileInputRef = useRef(null);

  useImperativeHandle(ref, () => ({
    openFileDialog: () => {
      if(fileInputRef.current) fileInputRef.current.click();
    }
  }));

  useEffect(() => {
    if(!file){
      setPreviewUrl(null);
      if(typeof onPreview === 'function') onPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    if(typeof onPreview === 'function') onPreview({ url, name: file.name });
    return () => { URL.revokeObjectURL(url); };
  }, [file]);

  function handleFiles(f){
    if(!f) return;
    const first = f[0];
    if(first.size > MAX_BYTES){
      alert(`File too large. Max ${(MAX_BYTES/1024/1024).toFixed(0)}MB.`);
      return;
    }
    setFile(first);
    // auto-submit when a file is selected through the external button
    // use a microtask so state updates can settle if needed
    setTimeout(() => submitFile(null, first), 80);
  }

  // submitFile can accept an explicit file (fileArg) to avoid stale state
  async function submitFile(e, fileArg){
    e && e.preventDefault();
    const f = fileArg || file;
    if(!f) return;
    setLoading(true);
    if(typeof onLoadingChange === 'function') onLoadingChange(true);
    const fd = new FormData();
    fd.append("file", f);
    if (model) fd.append("model", model);
    try{
      const res = await fetch(`${apiBase}/predict-file`, { method: "POST", body: fd });
      const json = await res.json();
      onResult && onResult(json);
    }catch(err){
      onResult && onResult({ error: err.toString() });
    }finally{ setLoading(false); if(typeof onLoadingChange === 'function') onLoadingChange(false); }
  }

  // local-path prediction removed; file upload is primary flow

  return (
    <div className="space-y-3">
      <form onSubmit={submitFile} aria-label="Upload image form">
        <label className="block text-sm font-medium mb-1">Upload image</label>

        <div
          ref={dropRef}
          onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
          onDrop={e => { e.preventDefault(); handleFiles(e.dataTransfer.files); }}
          className="border-2 border-dashed border-slate-200 rounded-lg p-4 flex items-center justify-between gap-4 panel-grad hover:border-sky-300 transition"
          role="button"
          tabIndex={0}
          onKeyDown={e => { if(e.key === 'Enter') { document.getElementById('file-input')?.click(); } }}
          aria-label="Drag and drop file here or browse"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded bg-slate-50 flex items-center justify-center">📁</div>
            <div>
              <div className="text-sm text-slate-600">Drag and drop file here</div>
              <div className="text-xs text-slate-400">Limit 100MB. Accepts PNG, JPG, TIFF</div>
            </div>
          </div>
          <div>
            <input ref={fileInputRef} id="file-input" type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" className="hidden" onChange={e => handleFiles(e.target.files)} />
            {/* Removed internal Browse button - external single Upload button will be used */}
          </div>
        </div>

        {file && (
          <div className="mt-3 flex items-center gap-4">
            {previewUrl ? (
              <img src={previewUrl} alt="preview" onError={() => { setPreviewUrl(null); if(typeof onPreview === 'function') onPreview(null); }} className="w-48 h-48 object-cover rounded border shadow-soft" />
            ) : (
              <div className="w-48 h-48 flex items-center justify-center rounded border panel-grad text-sm" style={{color:'#475569'}}>Preview not available</div>
            )}
            <div className="text-sm">
              <div className="font-medium text-lg">{file.name}</div>
              <div className="text-xs text-slate-400">{(file.size/1024/1024).toFixed(2)} MB</div>
            </div>
          </div>
        )}

        {/* No internal submit button - selection from the external Upload button triggers auto-submit */}
      </form>

      {/* Local server path input removed per request */}
    </div>
  );
}

export default forwardRef(UploadPanel);
