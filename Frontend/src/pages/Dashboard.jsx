import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import ImageUpload from '../components/ImageUpload';
import PredictionResult from '../components/PredictionResult';
import ModelDetails from '../components/ModelDetails';
import api from '../utils/api';
import '../index.css';

const Dashboard = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [selectedModel, setSelectedModel] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [stats, setStats] = useState({ total_scans: 0 });
    const [showDetails, setShowDetails] = useState(false);

    useEffect(() => {
        const fetchStats = async () => {
            const token = localStorage.getItem('token');
            if (!token) return;

            try {
                const response = await api.get('/users/me/stats');
                setStats(response.data);
            } catch (err) {
                console.error("Failed to fetch stats", err);
            }
        };
        fetchStats();
    }, [prediction]);

    const handleFileSelect = async (file) => {
        setSelectedFile(file);
        setPrediction(null);
        setError(null);

        const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');

        if (isTiff) {
            setPreviewUrl(null); // Clear previous preview
            // Fetch preview from backend
            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('http://localhost:8000/utils/convert-preview', {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    setPreviewUrl(url);
                } else {
                    console.error("Failed to fetch preview");
                }
            } catch (err) {
                console.error("Error fetching preview:", err);
            }
        } else {
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    const handlePredict = async () => {
        if (!selectedFile || !selectedModel) return;

        setLoading(true);
        setError(null);
        setPrediction(null);

        const token = localStorage.getItem('token');
        if (!token) {
            setError("Please login to use this feature");
            setLoading(false);
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            let endpoint = '';
            if (selectedModel === 'XGBoost') endpoint = '/predict/xgboost';
            else if (selectedModel === 'CNN') endpoint = '/predict/cnn';
            else if (selectedModel === 'Hybrid CNN') endpoint = '/predict/hybrid';

            const response = await api.post(endpoint, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            setPrediction(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Prediction failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="dashboard-container">
            <div className="dashboard-header">
                <h1>Printer Source Identification</h1>
                <div className="stats-container">
                    <span className="stat-item">Total Scans: <strong>{stats.total_scans}</strong></span>
                </div>
            </div>

            <div className="main-content">
                <div className="left-panel">
                    <ImageUpload
                        onFileSelect={handleFileSelect}
                        previewUrl={previewUrl}
                        selectedFile={selectedFile}
                    />
                </div>

                <div className="right-panel">
                    <div className="control-panel">
                        <h3>Analysis Controls</h3>

                        <div className="form-group">
                            <label htmlFor="model-select">Select AI Model:</label>
                            <select
                                id="model-select"
                                value={selectedModel || ''}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="model-dropdown"
                            >
                                <option value="" disabled>-- Choose a Model --</option>
                                <option value="XGBoost">XGBoost (47.67%)</option>
                                <option value="CNN">Simple CNN (88.15%)</option>
                                <option value="Hybrid CNN">Hybrid CNN (94.45%)</option>
                            </select>
                        </div>

                        <div className="button-group">
                            <button
                                className="btn-primary btn-predict"
                                onClick={handlePredict}
                                disabled={loading || !selectedFile || !selectedModel}
                            >
                                {loading ? 'Analyzing...' : 'Run Prediction'}
                            </button>

                            <button
                                className="btn-secondary btn-details"
                                onClick={() => setShowDetails(true)}
                            >
                                Model Details
                            </button>
                        </div>
                    </div>

                    {error && <div className="error-message">{error}</div>}

                    {prediction && (
                        <PredictionResult prediction={prediction} />
                    )}
                </div>
            </div>

            <ModelDetails isOpen={showDetails} onClose={() => setShowDetails(false)} />
        </div>
    );
};

export default Dashboard;
