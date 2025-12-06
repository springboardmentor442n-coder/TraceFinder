import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../index.css';

const History = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchHistory = async () => {
            const token = localStorage.getItem('token');
            if (!token) {
                setError('Not authenticated');
                setLoading(false);
                return;
            }

            try {
                const response = await fetch('http://localhost:8000/users/me/history', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (!response.ok) {
                    throw new Error('Failed to fetch history');
                }

                const data = await response.json();
                setHistory(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, []);

    return (
        <div className="container">
            <div className="header-section">
                <h1>Prediction History</h1>
                <Link to="/dashboard" className="btn-secondary">Back to Dashboard</Link>
            </div>

            {loading && <div className="loading-spinner"></div>}
            {error && <div className="error-message">{error}</div>}

            {!loading && !error && (
                <div className="history-table-container">
                    {history.length === 0 ? (
                        <p>No history found. Start scanning documents!</p>
                    ) : (
                        <table className="history-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>File Name</th>
                                    <th>Model</th>
                                    <th>Result</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {history.map((item) => (
                                    <tr key={item.id}>
                                        <td>{new Date(item.timestamp).toLocaleString()}</td>
                                        <td>{item.filename}</td>
                                        <td>{item.model_used}</td>
                                        <td>{item.prediction_result}</td>
                                        <td>{(item.confidence * 100).toFixed(2)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            )}
        </div>
    );
};

export default History;
