import React from 'react';
import '../index.css';

const LearnMore = () => {
    return (
        <div className="learn-more-container">
            <h1>How It Works</h1>

            <section className="info-section">
                <h2>Project Overview</h2>
                <p>
                    This project utilizes advanced machine learning techniques to identify the source printer
                    of a document based on microscopic artifacts left by the printing process.
                </p>
            </section>

            <section className="info-section">
                <h2>Our Models</h2>
                <div className="model-card">
                    <h3>XGBoost</h3>
                    <p>
                        Uses handcrafted features like pixel density, mean intensity, and edge density.
                        Fast and efficient for initial screening.
                    </p>
                </div>
                <div className="model-card">
                    <h3>CNN (Convolutional Neural Network)</h3>
                    <p>
                        A deep learning model trained on image residuals. It automatically learns
                        complex patterns and noise signatures specific to each printer.
                    </p>
                </div>
                <div className="model-card">
                    <h3>Hybrid CNN</h3>
                    <p>
                        Combines the power of deep learning (CNN) with handcrafted features.
                        This ensemble approach typically yields the highest accuracy by leveraging both
                        learned and engineered features.
                    </p>
                </div>
            </section>

            <section className="info-section">
                <h2>Workflow</h2>
                <ol>
                    <li><strong>Upload:</strong> User uploads a scanned document image.</li>
                    <li><strong>Preprocessing:</strong> Image is converted to grayscale, resized, and denoised to extract residuals.</li>
                    <li><strong>Feature Extraction:</strong> For XGBoost/Hybrid, statistical features are calculated.</li>
                    <li><strong>Prediction:</strong> The selected model analyzes the data and predicts the source printer.</li>
                </ol>
            </section>
        </div>
    );
};

export default LearnMore;
