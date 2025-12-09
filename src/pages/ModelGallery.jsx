import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

/*
  Shows three model cards (CNN, Hybrid CNN, Sklearn)
  Modal to pick a model (B: use modal to pick model)
*/

const MODELS = [
  { id: "cnn", title: "CNN model", accuracy: "93%", image: "/assets/cnn_confusionmatrix.png" },
  { id: "hybrid", title: "Hybrid CNN", accuracy: "83%", image: "/assets/hybrid_cnn_confusionmatrix.png" },
  { id: "sklearn", title: "Sklearn model", accuracy: "46%", image: "/assets/sklearn_confusionmatrix.png" }
];

export default function ModelGallery(){
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState(null);
  const navigate = useNavigate();

  function openPicker(model){
    setSelected(model);
    setOpen(true);
  }
  function goModel(){
    if(selected) {
      navigate(`/models/${selected.id}`);
    }
  }

  return (
    <div className="page-bg">
      <div className="page-frame">
        <div style={{minHeight:'calc(100vh - 8rem)', display:'flex', flexDirection:'column', justifyContent:'center'}}>
          <div style={{margin: '1rem 0'}}>
            <button className="back-small" onClick={() => window.history.back()}>Back</button>
          </div>

          <div className="text-center mb-8">
            <div className="title-box fade-in">
              <h2 className="text-3xl font-bold" style={{letterSpacing:'.12em'}}>Select models</h2>
            </div>
          </div>

          <div className="gallery-3" style={{alignItems:'stretch'}}>
          {MODELS.map(m => (
            <div key={m.id} className="model-card--sketch flex flex-col justify-center items-center text-center fade-in" onClick={() => navigate(`/models/${m.id}`)} style={{height:320}}>
              <div>
                <div className="model-name">{m.title}</div>
                <div style={{marginTop:0, fontSize:14, color:'#334155'}}>accuracy: <span className="accuracy">{m.accuracy}</span></div>
              </div>
            </div>
          ))}
          </div>
        </div>
      </div>
    </div>
  );
}
