import axios from 'axios';

// Base API URL - use environment variable if available, otherwise default to localhost
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Upload an image for object detection
 *
 * @param {File} file - Image file to upload
 * @param {number} confidenceThreshold - Confidence threshold (0-1)
 * @param {number} maxDetections - Maximum number of detections to return
 * @returns {Promise<Object>} - Detection results
 */
export const uploadImage = async (file, confidenceThreshold = 0.25, maxDetections = 100) => {
  try {
    // Create FormData object
    const formData = new FormData();
    formData.append('file', file);

    // Add query parameters
    const params = new URLSearchParams();
    params.append('conf_threshold', confidenceThreshold);
    params.append('max_detections', maxDetections);

    // Send request
    const response = await axios.post(
      `${API_URL}/predict?${params.toString()}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      }
    );

    return response.data;
  } catch (error) {
    console.error('API Error:', error);

    // Handle different error types
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      const message = error.response.data?.detail || `Error ${error.response.status}: ${error.response.statusText}`;
      throw new Error(message);
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('No response from server. Please check your connection and try again.');
    } else {
      // Something happened in setting up the request
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

/**
 * Upload multiple images for batch processing
 *
 * @param {File[]} files - Array of image files to upload
 * @param {number} confidenceThreshold - Confidence threshold (0-1)
 * @param {number} maxDetections - Maximum number of detections to return
 * @returns {Promise<Object>} - Batch detection results
 */
export const uploadBatchImages = async (files, confidenceThreshold = 0.25, maxDetections = 100) => {
  try {
    // Create FormData object
    const formData = new FormData();

    // Append each file with the same field name
    files.forEach(file => {
      formData.append('files', file);
    });

    // Add query parameters
    const params = new URLSearchParams();
    params.append('conf_threshold', confidenceThreshold);
    params.append('max_detections', maxDetections);

    // Send request
    const response = await axios.post(
      `${API_URL}/batch_predict?${params.toString()}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 second timeout for batch processing
      }
    );

    return response.data;
  } catch (error) {
    console.error('API Error (Batch):', error);

    // Handle different error types
    if (error.response) {
      const message = error.response.data?.detail || `Error ${error.response.status}: ${error.response.statusText}`;
      throw new Error(message);
    } else if (error.request) {
      throw new Error('No response from server. Please check your connection and try again.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
};

/**
 * Check API health
 *
 * @returns {Promise<Object>} - Health check result
 */
export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_URL}/health`, {
      timeout: 5000, // 5 second timeout
    });

    return response.data;
  } catch (error) {
    console.error('Health Check Error:', error);
    throw new Error('API service is not available');
  }
};

/**
 * Get model information
 *
 * @returns {Promise<Object>} - Model information
 */
export const getModelInfo = async () => {
  try {
    const response = await axios.get(`${API_URL}/model`, {
      timeout: 5000,
    });

    return response.data;
  } catch (error) {
    console.error('Model Info Error:', error);
    throw new Error('Could not retrieve model information');
  }
};

export default {
  uploadImage,
  uploadBatchImages,
  checkHealth,
  getModelInfo,
};