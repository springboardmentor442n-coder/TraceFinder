import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import '../index.css';

const Navbar = () => {
    const navigate = useNavigate();
    const token = localStorage.getItem('token');

    const handleLogout = () => {
        localStorage.removeItem('token');
        navigate('/login');
    };

    return (
        <nav className="navbar">
            <div className="navbar-brand">
                <Link to="/">PrinterID</Link>
            </div>
            <div className="navbar-links">
                <Link to="/">Home</Link>
                <Link to="/learn-more">Learn More</Link>
                {token ? (
                    <>

                        <button onClick={handleLogout} className="btn-logout">Logout</button>
                    </>
                ) : (
                    <>
                        <Link to="/login">Login</Link>
                        <Link to="/signup" className="btn-signup">Sign Up</Link>
                    </>
                )}
            </div>
        </nav>
    );
};

export default Navbar;
