import React, { useState } from "react";
import { useParams, Link } from "react-router-dom";
// using static confusion images per user's preference (no dynamic grid component)

const META = {
  cnn: {
    id: 'cnn',
    title: "Convolutional Neural Network (CNN)",
    desc: "Image-only CNN using residuals (256×256) created from grayscale, normalized, wavelet-denoised images.",
    summary: "Residual-based CNN that predicts scanner model from image patterns.",
    accuracy: "93%",
    trainingImages: 4568,
    preprocessing: [
      'Grayscale conversion',
      'Resize → 256×256',
      'Normalize [0–1]',
      'Haar wavelet denoising',
      'Residual creation',
      'Reshape to 256×256×1'
    ],
    confusion: "/assets/cnn_confusionmatrix.png"
  },

  hybrid: {
    id: 'hybrid',
    title: "Hybrid CNN",
    desc: "Dual-input model combining residual image + 31 handcrafted statistical features.",
    summary: "CNN branch + feature branch fused for more robust scanner prediction.",
    accuracy: "83%",
    trainingImages: 5200,
    preprocessing: [
      'Grayscale conversion',
      'Resize → 256×256',
      'Residual creation',
      '31-feature extraction (LBP, FFT, GLCM, correlations)',
      'Normalize features'
    ],
    confusion: "/assets/hybrid_cnn_confusionmatrix.png"
  },

  sklearn: {
    id: 'sklearn',
    title: "Sklearn classifier",
    desc: "Classical ML classifier trained on handcrafted statistical features only.",
    summary: "Fast baseline model using handcrafted features without CNN.",
    accuracy: "46%",
    trainingImages: 4000,
    preprocessing: [
      'Grayscale conversion',
      'Resize (for feature extraction)',
      'Compute handcrafted features',
      'Normalize features'
    ],
    confusion: "/assets/sklearn_confusionmatrix.png"
  }
};


export default function ModelDetail(){
  const { modelId } = useParams();
  const info = META[modelId] || META.cnn;
  const [result, setResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [running, setRunning] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);

  const sampleLabels = ['Canon120-1','Canon220','Canon9000-1','EpsonV370-1','HP'];

  function handleFileChange(e){
    const f = e.target.files && e.target.files[0];
    if(!f) return setSelectedFile(null);
    setSelectedFile(f);
    try{ setPreviewUrl(URL.createObjectURL(f)); } catch(e){ setPreviewUrl(null); }
    setLastPrediction(null);
  }

  function runMockPrediction(){
    if(!selectedFile) return;
    setRunning(true);
    setLastPrediction(null);
    // simulate prediction
    setTimeout(()=>{
      const label = sampleLabels[Math.floor(Math.random()*sampleLabels.length)];
      const confidence = (60 + Math.floor(Math.random()*40));
      const res = { label, confidence };
      setLastPrediction(res);
      setRunning(false);
      setResult(res);
    }, 700 + Math.floor(Math.random()*600));
  }

  return (
    <div className="page-bg">
      <div className="page-frame">
        <div style={{textAlign:'center', marginTop:8}}>
          <div className="title-box" style={{display:'inline-block'}}>
            <strong>{info.title}</strong>
          </div>
        </div>

        <div style={{height:18}} />

        <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:12, marginBottom:12}}>
          <div>
            <button className="back-small" onClick={() => window.history.back()}>Back</button>
          </div>
        </div>

        <div>
          <div className="two-col" style={{marginTop:18}}>
              <div className="left-panel" style={{display:'flex', flexDirection:'column', gap:14, alignItems:'flex-start'}}>
              <div className="confusion-caption">Confusion matrix</div>
              <div className="confusion-wrap" style={{width:'100%', maxWidth:780}}>
                <img src={info.confusion} alt="confusion" className="confusion-img" style={{maxWidth:760, maxHeight:520}} />
              </div>

              <div style={{marginTop:12}}>
                <div className="upload-row">
                  <div style={{display:'flex', flexDirection:'column', gap:12, flex: '1 1 auto'}}>
                    <div className="upload-bar upload-large teal">
                      <div className="file-info" style={{flex: '1 1 auto'}}>
                        <div className="file-thumb" style={{display:'flex',alignItems:'center',justifyContent:'center',color:'#64748b', fontWeight:700}}>thumb</div>
                        <div style={{minWidth:0}}>
                          <div style={{fontWeight:800, fontSize:16}}>{selectedFile?.name ?? 'No file selected'}</div>
                          <div style={{fontSize:13, color:'#475569'}}>supported: jpg, png, tiff</div>
                        </div>
                      </div>
                      <div style={{flex: '0 0 auto'}}>
                        <label className="upload-btn">
                          <input type="file" accept="image/*" style={{display:'none'}} onChange={handleFileChange} />
                          upload file
                        </label>
                      </div>
                    </div>

                    <div style={{display:'flex', justifyContent:'flex-start', alignItems:'center', gap:12}}>
                      <button className="upload-btn run-btn" disabled={!selectedFile || running} onClick={runMockPrediction} style={{minWidth:160}}>
                        {running ? (
                          <span style={{display:'inline-flex',alignItems:'center',gap:8}}>
                            <span className="spinner-small" aria-hidden>
                              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2v4" stroke="#0b2540" strokeWidth="2" strokeLinecap="round"/>
                                <path d="M20.5 7.5l-3 3" stroke="#0b2540" strokeWidth="2" strokeLinecap="round"/>
                                <path d="M22 12a10 10 0 11-10-10" stroke="#0b2540" strokeWidth="2" strokeLinecap="round"/>
                              </svg>
                            </span>
                            Running...
                          </span>
                        ) : 'Run Prediction'}
                      </button>
                      {running && <span className="spinner" aria-hidden="true" />}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div style={{display:'flex', flexDirection:'column', gap:16}}>
              <div className="panel-grad">
                <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <h3 className="font-semibold">Model Details</h3>
                  <div className="text-xs">Summary</div>
                </div>
                <div style={{marginTop:12}}>
                  <div style={{display:'flex', justifyContent:'space-between'}}>
                    <div>Model Accuracy:</div>
                    <div>{info.accuracy}</div>
                  </div>
                  <div style={{display:'flex', justifyContent:'space-between', marginTop:6}}>
                    <div>Training images:</div>
                    <div>{info.trainingImages.toLocaleString?.() ?? info.trainingImages}</div>
                  </div>
                  <div style={{marginTop:8}}>
                    <div style={{fontWeight:700}}>Model Description:</div>
                    <div style={{marginTop:6}}>{info.desc}</div>
                  </div>
                </div>
              </div>

              <div className="panel-grad">
                <div style={{fontWeight:700}}>Preprocessing Pipeline:</div>
                <ul style={{marginTop:8, paddingLeft:18}}>
                  {info.preprocessing.map((p,i)=>(<li key={i}>{p}</li>))}
                </ul>
              </div>

              <div className="result-small warm" style={{width:'100%'}}>
                {lastPrediction ? (
                  <div>
                    <div style={{fontWeight:700}}>Prediction</div>
                    <div style={{fontSize:18, marginTop:6}}>{lastPrediction.label} <span style={{fontWeight:600, marginLeft:8}}>{lastPrediction.confidence}%</span></div>
                  </div>
                ) : (
                  <div style={{color:'#475569'}}>No result yet</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
