import React from "react";
import { Link } from "react-router-dom";

export default function Home(){
  return (
    <div className="page-bg">
      <div className="page-frame">
        <div className="home-hero fade-in">
          <h1 className="hero-title-text" style={{fontSize: '96px'}}>TRACE FINDER</h1>
          <div style={{height:18}} />
          <p className="text-lg text-slate-700" style={{opacity:0.95}}>Scanner-source identification</p>
          <p style={{maxWidth:780, margin:'1rem auto', color:'#475569'}}>Upload a scanned image to predict which scanner or device produced it. Trace Finder analyses residual patterns and handcrafted features to identify device sources with high accuracy.</p>

          <div style={{height:18}} />
          <Link to="/models" className="cta-sketched">explore models</Link>
        </div>
      </div>
    </div>
  )
}
