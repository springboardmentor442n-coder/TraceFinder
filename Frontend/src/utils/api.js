import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
});

// Request interceptor to add auth token
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor to handle 401 (Token expired/invalid)
api.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response && error.response.status === 401) {
            // Token expired or invalid - logout user
            localStorage.removeItem('token');

            // Show session expired message
            if (!window.location.pathname.includes('/login')) {
                alert('Your session has expired. Please login again.');
                window.location.href = '/login';
            }
        }
        return Promise.reject(error);
    }
);

export default api;
