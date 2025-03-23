import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadImage } from '../services/detectionApi';

const Home = () => {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [threshold, setThreshold] = useState(0.25);
  const [maxDetections, setMaxDetections] = useState(100);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  // Clear selected file and URL when component unmounts
  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (selectedFile) {
      // Check if file is an image
      if (!selectedFile.type.match('image.*')) {
        setError('Please select an image file (JPEG, PNG, etc.)');
        setFile(null);
        setImageUrl(null);
        return;
      }

      // Check file size (limit to 10MB)
      if (selectedFile.size > 10 * 1024 * 1024) {
        setError('File size too large. Please select an image under 10MB.');
        setFile(null);
        setImageUrl(null);
        return;
      }

      setFile(selectedFile);
      setImageUrl(URL.createObjectURL(selectedFile));
      setError(null);
    } else {
      setFile(null);
      setImageUrl(null);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Please select an image file');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await uploadImage(file, threshold, maxDetections);

      // Navigate to results page with response data
      navigate(`/results/${Date.now()}`, { state: { results: response, imageUrl } });
    } catch (error) {
      console.error('Error detecting objects:', error);
      setError(error.message || 'Error detecting objects. Please try again.');
      setIsLoading(false);
    }
  };

  return (
    <div className="home-page">
      <div className="card">
        <h2 className="card-title">Object Detection</h2>
        <p>Upload an image to detect and classify objects using our deep learning model.</p>

        {error && <div className="status-message error-message">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="two-column">
            <div className="control-panel">
              <div className="form-group">
                <label htmlFor="image-upload">Select an image</label>
                <div
                  className={`file-input-button ${file ? 'has-file' : ''}`}
                  onClick={() => fileInputRef.current.click()}
                >
                  {file ? (
                    <>
                      <div>Selected: {file.name}</div>
                      <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </div>
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                      </svg>
                      <div style={{ marginTop: '0.5rem' }}>
                        Click to browse or drag an image here
                      </div>
                      <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', color: '#666' }}>
                        Supported formats: JPG, PNG, WEBP
                      </div>
                    </>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    id="image-upload"
                    accept="image/*"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="threshold">Confidence Threshold: {threshold}</label>
                <input
                  type="range"
                  id="threshold"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                />
                <div className="range-labels">
                  <span>0.05</span>
                  <span>0.50</span>
                  <span>0.95</span>
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="max-detections">Max Detections: {maxDetections}</label>
                <input
                  type="range"
                  id="max-detections"
                  min="1"
                  max="200"
                  step="1"
                  value={maxDetections}
                  onChange={(e) => setMaxDetections(parseInt(e.target.value))}
                />
                <div className="range-labels">
                  <span>1</span>
                  <span>100</span>
                  <span>200</span>
                </div>
              </div>

              <button type="submit" disabled={!file || isLoading}>
                {isLoading ? 'Detecting...' : 'Detect Objects'}
              </button>
            </div>

            <div className="preview-panel">
              {isLoading ? (
                <div className="loading-container">
                  <div className="loading-spinner"></div>
                  <p>Processing image...</p>
                </div>
              ) : imageUrl ? (
                <div className="image-preview">
                  <img src={imageUrl} alt="Preview" />
                </div>
              ) : (
                <div className="image-preview">
                  <p>Upload an image to see a preview</p>
                </div>
              )}
            </div>
          </div>
        </form>
      </div>

      <div className="card">
        <h2 className="card-title">About This Tool</h2>
        <p>This object detection system uses a YOLO (You Only Look Once) architecture to detect and classify objects in images.</p>
        <ul style={{ marginTop: '1rem', marginLeft: '1.5rem' }}>
          <li>Based on YOLOv5 model architecture</li>
          <li>Trained on the COCO dataset with 80 common object classes</li>
          <li>Optimized for both accuracy and inference speed</li>
          <li>Adjust the confidence threshold to filter detections</li>
        </ul>
      </div>
    </div>
  );
};

export default Home;