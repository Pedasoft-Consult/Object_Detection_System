import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
// eslint-disable-next-line no-unused-vars
import { formatTime } from '../utils/helpers';


const DetectionResults = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);

  useEffect(() => {
    // Check if we have results in the location state
    if (location.state?.results && location.state?.imageUrl) {
      setResults(location.state.results);
      setImageUrl(location.state.imageUrl);
    } else {
      // If no results, redirect to home
      navigate('/');
    }

    // Clean up URL object when component unmounts
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [location, navigate, imageUrl]);

  // Group detections by class
  const getDetectionsByClass = () => {
    if (!results?.detections) return {};

    return results.detections.reduce((acc, detection) => {
      const className = detection.class_name;
      if (!acc[className]) {
        acc[className] = [];
      }
      acc[className].push(detection);
      return acc;
    }, {});
  };

  // Calculate statistics
  const getStatistics = () => {
    if (!results?.detections) return {};

    const detectionsByClass = getDetectionsByClass();
    const numClasses = Object.keys(detectionsByClass).length;
    const highestConfidence = results.detections.length > 0 ?
      Math.max(...results.detections.map(d => d.confidence)) : 0;

    return {
      totalDetections: results.detections.length,
      numClasses,
      highestConfidence,
      inferenceTime: results.inference_time,
      imageSize: results.image_size,
    };
  };

  if (!results) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading results...</p>
      </div>
    );
  }

  const detectionsByClass = getDetectionsByClass();
  const statistics = getStatistics();

  return (
    <div className="detection-results-page">
      <div className="card">
        <div className="detection-header">
          <h2 className="card-title">Detection Results</h2>
          <div className="inference-time">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            Inference time: {(statistics.inferenceTime * 1000).toFixed(2)} ms
          </div>
        </div>

        <div className="detection-container">
          <div className="result-visualization">
            {imageUrl && (
              <img src={imageUrl} alt="Uploaded" className="result-image" />
            )}
          </div>

          <div className="result-sidebar">
            <div className="statistics-panel">
              <h3>Statistics</h3>
              <div className="statistics-item">
                <span className="statistics-label">Total Detections:</span>
                <span className="statistics-value">{statistics.totalDetections}</span>
              </div>
              <div className="statistics-item">
                <span className="statistics-label">Classes Detected:</span>
                <span className="statistics-value">{statistics.numClasses}</span>
              </div>
              <div className="statistics-item">
                <span className="statistics-label">Highest Confidence:</span>
                <span className="statistics-value">{(statistics.highestConfidence * 100).toFixed(1)}%</span>
              </div>
              <div className="statistics-item">
                <span className="statistics-label">Image Size:</span>
                <span className="statistics-value">{statistics.imageSize?.[0]} × {statistics.imageSize?.[1]}</span>
              </div>
            </div>

            <div className="detection-list-container">
              <h3>Detections</h3>
              {Object.keys(detectionsByClass).length > 0 ? (
                Object.entries(detectionsByClass).map(([className, detections]) => (
                  <div key={className} className="detection-class-group">
                    <h4 className="detection-class-heading">
                      {className} ({detections.length})
                    </h4>
                    <div className="detection-items">
                      {detections.map((detection, index) => (
                        <div key={index} className="detection-item">
                          <span className="detection-class">{detection.class_name}</span>
                          <span className="detection-confidence">{(detection.confidence * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <div className="no-detections">No objects detected</div>
              )}
            </div>
          </div>
        </div>

        <div className="action-buttons" style={{ marginTop: '2rem' }}>
          <Link to="/" className="button">Back to Home</Link>
          <button
            onClick={() => navigate('/')}
            className="button"
            style={{ marginLeft: '1rem', background: 'var(--primary-color)' }}
          >
            Detect Another Image
          </button>
        </div>
      </div>
    </div>
  );
};

export default DetectionResults;