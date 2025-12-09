import React from "react";
import { useNavigate } from "react-router-dom";

export default function ModelPickerModal({ model, onClose }){
  const nav = useNavigate();
  if(!model) return null;
  const modelId = model.id || model.key || "cnn";
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="panel-grad rounded-lg p-6 z-50 w-full max-w-lg shadow-lg">
        <h3 className="text-xl font-semibold mb-2">Open model</h3>
        <p className="text-sm text-slate-600 mb-4">{model.title}</p>
        <div className="flex gap-3 justify-end">
          <button className="px-4 py-2" onClick={onClose}>Close</button>
          <button
            className="px-4 py-2 bg-indigo-600 text-white rounded"
            onClick={() => { onClose && onClose(); nav(`/models/${modelId}`); }}
          >
            Open
          </button>
        </div>
      </div>
    </div>
  );
}
