import React from 'react';
import { Link } from 'react-router-dom';
import '../index.css';

const LandingPage = () => {
    return (
        <div className="landing-container">
            <div className="hero-section">
                <h1 className="hero-title">Printer Source Identification</h1>
                <p className="hero-description">
                    Uncover the origin of your documents with advanced AI forensics.
                    Detect the source printer of any image with high precision using our
                    XGBoost, CNN, and Hybrid models.
                </p>
                <div className="hero-buttons">
                    <Link to="/dashboard" className="btn-primary">Start with Us</Link>
                    <Link to="/learn-more" className="btn-secondary">Learn More</Link>
                </div>
            </div>
        </div>
    );
};

export default LandingPage;
