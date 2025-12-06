import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const ImageUpload = ({ onFileSelect, previewUrl, selectedFile }) => {
    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles && acceptedFiles.length > 0) {
            onFileSelect(acceptedFiles[0]);
        }
    }, [onFileSelect]);

    const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png', '.tif', '.tiff']
        },
        multiple: false,
        noClick: false // Enable click on container
    });

    return (
        <div className="image-upload-container">
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />

                {previewUrl ? (
                    <div className="image-preview-container">
                        <img
                            src={previewUrl}
                            alt="Preview"
                            className="preview-image"
                            onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = 'https://via.placeholder.com/400x300?text=Preview+Error';
                            }}
                        />
                        <div className="preview-actions">
                            <button className="btn-choose-file" onClick={open}>Change File</button>
                        </div>
                    </div>
                ) : (
                    <div className="upload-placeholder">
                        <div className="folder-icon">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M20 6H12L10 4H4C2.89543 4 2 4.89543 2 6V18C2 19.1046 2.89543 20 4 20H20C21.1046 20 22 19.1046 22 18V8C22 6.89543 21.1046 6 20 6Z" fill="#4F46E5" />
                            </svg>
                        </div>
                        <p className="upload-text">Click the button below to upload your files.</p>

                        <div className="upload-separator">
                            <span className="separator-line"></span>
                            <span className="separator-text">OR</span>
                            <span className="separator-line"></span>
                        </div>

                        <button className="btn-choose-file">
                            Choose File
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageUpload;
