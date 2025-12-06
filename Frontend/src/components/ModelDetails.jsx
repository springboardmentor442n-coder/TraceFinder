import React, { useState, useEffect } from 'react';

const ModelDetails = ({ isOpen, onClose }) => {
    const [activeTab, setActiveTab] = useState('xgboost');
    const [history, setHistory] = useState([]);

    useEffect(() => {
        if (isOpen) {
            fetchHistory();
        }
    }, [isOpen]);

    const fetchHistory = async () => {
        const token = localStorage.getItem('token');
        if (!token) return;

        try {
            const response = await fetch('http://localhost:8000/users/me/history', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setHistory(data);
            }
        } catch (err) {
            console.error("Failed to fetch history", err);
        }
    };

    const downloadHistory = () => {
        if (history.length === 0) return;

        const headers = ["Filename", "Model", "Prediction", "Confidence", "Timestamp"];
        const csvContent = [
            headers.join(","),
            ...history.map(row => [
                row.filename,
                row.model_used,
                row.prediction_result,
                (row.confidence * 100).toFixed(2) + "%",
                new Date(row.timestamp).toLocaleString()
            ].join(","))
        ].join("\n");

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "prediction_history.csv");
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <div className="modal-header">
                    <h2>Model Details & Architecture</h2>
                    <button className="close-btn" onClick={onClose}>&times;</button>
                </div>

                <div className="tabs">
                    <button className={`tab-btn ${activeTab === 'xgboost' ? 'active' : ''}`} onClick={() => setActiveTab('xgboost')}>XGBoost</button>
                    <button className={`tab-btn ${activeTab === 'cnn' ? 'active' : ''}`} onClick={() => setActiveTab('cnn')}>Simple CNN</button>
                    <button className={`tab-btn ${activeTab === 'hybrid' ? 'active' : ''}`} onClick={() => setActiveTab('hybrid')}>Hybrid CNN</button>
                    <button className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`} onClick={() => setActiveTab('logs')}>Prediction Logs</button>
                </div>

                <div className="tab-content">
                    {activeTab === 'xgboost' && (
                        <div className="model-info">
                            <h3>XGBoost (eXtreme Gradient Boosting)</h3>
                            <p>A powerful machine learning algorithm that uses gradient boosting framework. It extracts statistical features from images (like GLCM, LBP) to classify printer sources.</p>
                            <div className="stats-grid">
                                <div className="stat-box">
                                    <span className="label">Accuracy</span>
                                    <span className="value">47.67%</span>
                                </div>
                                <div className="stat-box">
                                    <span className="label">Speed</span>
                                    <span className="value">Fast</span>
                                </div>
                            </div>
                            <div className="confusion-matrix">
                                <h4>Confusion Matrix</h4>
                                <img src="/assets/images/xgboost_cm.png" alt="XGBoost Confusion Matrix" />
                            </div>
                        </div>
                    )}

                    {activeTab === 'cnn' && (
                        <div className="model-info">
                            <h3>Simple CNN (Convolutional Neural Network)</h3>
                            <p>A deep learning model designed to process pixel data directly. It automatically learns spatial hierarchies of features from the input images.</p>

                            <div className="stats-grid">
                                <div className="stat-box">
                                    <span className="label">Accuracy</span>
                                    <span className="value">88.15%</span>
                                </div>
                                <div className="stat-box">
                                    <span className="label">Type</span>
                                    <span className="value">Sequential</span>
                                </div>
                            </div>

                            <div className="architecture-details">
                                <h4>Model Architecture: "sequential_2"</h4>
                                <pre className="code-block">
                                    {`Layer (type)                    Output Shape           Param #
=================================================================
conv2d_6 (Conv2D)               (None, 256, 256, 32)   320
max_pooling2d_6 (MaxPooling2D)  (None, 128, 128, 32)   0
conv2d_7 (Conv2D)               (None, 128, 128, 64)   18,496
max_pooling2d_7 (MaxPooling2D)  (None, 64, 64, 64)     0
conv2d_8 (Conv2D)               (None, 64, 64, 128)    73,856
max_pooling2d_8 (MaxPooling2D)  (None, 32, 32, 128)    0
flatten_2 (Flatten)             (None, 131072)         0
dense_4 (Dense)                 (None, 128)            16,777,344
dropout_2 (Dropout)             (None, 128)            0
dense_5 (Dense)                 (None, 11)             1,419
=================================================================
Total params: 16,871,435 (64.36 MB)`}
                                </pre>
                            </div>

                            <div className="confusion-matrix">
                                <h4>Confusion Matrix</h4>
                                <img src="/assets/images/cnn_cm.png" alt="CNN Confusion Matrix" />
                            </div>
                        </div>
                    )}

                    {activeTab === 'hybrid' && (
                        <div className="model-info">
                            <h3>Hybrid CNN (Best Performance)</h3>
                            <p>Combines the power of CNNs with handcrafted features. It uses both raw pixel data and extracted statistical features to achieve the highest accuracy.</p>

                            <div className="stats-grid">
                                <div className="stat-box">
                                    <span className="label">Accuracy</span>
                                    <span className="value">94.45%</span>
                                </div>
                                <div className="stat-box">
                                    <span className="label">Robustness</span>
                                    <span className="value">High</span>
                                </div>
                            </div>

                            <div className="architecture-details">
                                <h4>Model Architecture: "functional" (Anti-Overfitting)</h4>
                                <pre className="code-block">
                                    {`Layer (type)        Output Shape      Param #    Connected to
=================================================================
input_layer         (None, 256, 256,  0          -
(InputLayer)        1)
cast (Cast)         (None, 256, 256,  0          input_layer[0][0]
                    1)
conv_initial        (None, 256, 256,  320        cast[0][0]
(Conv2D)            32)
pool_initial        (None, 128, 128,  0          conv_initial[0][0]
(MaxPooling2D)      32)
hybrid_block_1_bra… (None, 128, 128,  1,056      pool_initial[0][0]
(Conv2D)            32)
hybrid_block_1_bra… (None, 128, 128,  9,248      pool_initial[0][0]
(Conv2D)            32)
hybrid_block_1_con… (None, 128, 128,  0          hybrid_block_1_br…
(Concatenate)       64)                          hybrid_block_1_br…
hybrid_block_1_pool (None, 64, 64,    0          hybrid_block_1_co…
(MaxPooling2D)      64)
conv_final (Conv2D) (None, 64, 64,    73,856     hybrid_block_1_po…
                    128)
pool_final          (None, 32, 32,    0          conv_final[0][0]
(MaxPooling2D)      128)
flatten (Flatten)   (None, 131072)    0          pool_final[0][0]
dense_1 (Dense)     (None, 128)       16,777,344 flatten[0][0]
dropout (Dropout)   (None, 128)       0          dense_1[0][0]
output (Dense)      (None, 11)        1,419      dropout[0][0]
=================================================================
Total params: 16,863,243 (64.33 MB)`}
                                </pre>
                            </div>

                            <div className="dataflow-section">
                                <h4>Dataflow & Process</h4>
                                <ol>
                                    <li><strong>Upload:</strong> User uploads a document image (supports TIFF, PNG, JPG).</li>
                                    <li><strong>Preprocessing:</strong> Image is resized to 256x256, converted to grayscale, and normalized. Residual noise is extracted.</li>
                                    <li><strong>Feature Extraction:</strong> Statistical features (GLCM, LBP) are computed.</li>
                                    <li><strong>Hybrid Processing:</strong> The CNN processes the image while dense layers process the statistical features.</li>
                                    <li><strong>Prediction:</strong> Both paths merge to classify the printer source with a confidence score.</li>
                                </ol>
                            </div>

                            <div className="confusion-matrix">
                                <h4>Confusion Matrix</h4>
                                <img src="/assets/images/hybrid_cm.png" alt="Hybrid CNN Confusion Matrix" />
                            </div>
                        </div>
                    )}

                    {activeTab === 'logs' && (
                        <div className="logs-view">
                            <div className="logs-header">
                                <h3>Recent Predictions</h3>
                                <button className="btn-secondary" onClick={downloadHistory}>Download CSV</button>
                            </div>
                            <div className="table-container">
                                <table className="history-table">
                                    <thead>
                                        <tr>
                                            <th>Date</th>
                                            <th>Filename</th>
                                            <th>Model</th>
                                            <th>Prediction</th>
                                            <th>Confidence</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {history.map((item, index) => (
                                            <tr key={index}>
                                                <td>{new Date(item.timestamp).toLocaleDateString()}</td>
                                                <td>{item.filename}</td>
                                                <td>{item.model_used}</td>
                                                <td>{item.prediction_result}</td>
                                                <td>
                                                    <span className="confidence-badge" style={{ opacity: item.confidence }}>
                                                        {(item.confidence * 100).toFixed(1)}%
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ModelDetails;
