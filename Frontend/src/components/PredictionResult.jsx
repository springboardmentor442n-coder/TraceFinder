import React from 'react';

const PredictionResult = ({ prediction }) => {
    const { model, prediction: label, confidence } = prediction;
    const confidencePercent = (confidence * 100).toFixed(2);

    return (
        <div className="prediction-result">
            <div className="result-header">
                <div className="success-icon">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM10 17L5 12L6.41 10.59L10 14.17L17.59 6.58L19 8L10 17Z" fill="#10B981" />
                    </svg>
                </div>
                <h3>Analysis Complete</h3>
            </div>

            <div className="result-card">
                <div className="result-row">
                    <span className="result-label">Model Used</span>
                    <span className="result-value">{model}</span>
                </div>

                <div className="result-divider"></div>

                <div className="result-row highlight">
                    <span className="result-label">Predicted Source</span>
                    <span className="result-value source-name">{label}</span>
                </div>

                <div className="result-divider"></div>

                <div className="result-row vertical">
                    <div className="confidence-header">
                        <span className="result-label">Confidence Score</span>
                        <span className="confidence-value">{confidencePercent}%</span>
                    </div>
                    <div className="confidence-track">
                        <div
                            className="confidence-fill"
                            style={{ width: `${confidencePercent}%` }}
                        ></div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PredictionResult;
