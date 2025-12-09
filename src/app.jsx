import React from "react";
import { Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import ModelGallery from "./pages/ModelGallery";
import ModelDetail from "./pages/ModelDetail";

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="p-4 flex items-center justify-between">
        <div className="text-lg font-semibold">&nbsp;</div>
        <div className="text-center flex-1">{/* header intentionally blank per design */}</div>
        <div style={{width:32}} />
      </header>

      <main className="px-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/models" element={<ModelGallery />} />
          <Route path="/models/:modelId" element={<ModelDetail />} />
        </Routes>
      </main>
    </div>
  );
}
