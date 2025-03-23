/**
 * Format file size into human-readable format
 *
 * @param {number} bytes - File size in bytes
 * @param {number} decimals - Number of decimal places
 * @returns {string} - Formatted file size (e.g., "2.5 MB")
 */
export const formatFileSize = (bytes, decimals = 2) => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

/**
 * Format time in milliseconds to human-readable format
 *
 * @param {number} ms - Time in milliseconds
 * @returns {string} - Formatted time
 */
export const formatTime = (ms) => {
  if (ms < 1000) {
    return `${ms.toFixed(2)} ms`;
  } else {
    return `${(ms / 1000).toFixed(2)} s`;
  }
};

/**
 * Get a color for a specific class ID
 *
 * @param {number|string} classId - Class ID or name
 * @returns {string} - Hex color code
 */
export const getClassColor = (classId) => {
  // Simple hash function for strings
  if (typeof classId === 'string') {
    let hash = 0;
    for (let i = 0; i < classId.length; i++) {
      hash = classId.charCodeAt(i) + ((hash << 5) - hash);
    }

    // Convert to hex color
    let color = '#';
    for (let i = 0; i < 3; i++) {
      const value = (hash >> (i * 8)) & 0xFF;
      color += ('00' + value.toString(16)).substr(-2);
    }
    return color;
  }

  // Color palette for numeric IDs
  const colors = [
    '#e74c3c', // Red
    '#3498db', // Blue
    '#2ecc71', // Green
    '#f39c12', // Orange
    '#9b59b6', // Purple
    '#1abc9c', // Teal
    '#d35400', // Pumpkin
    '#2980b9', // Dark Blue
    '#27ae60', // Dark Green
    '#e67e22', // Dark Orange
    '#8e44ad', // Dark Purple
    '#16a085', // Dark Teal
    '#c0392b', // Dark Red
    '#7f8c8d', // Gray
    '#f1c40f', // Yellow
    '#34495e', // Navy
  ];

  return colors[classId % colors.length];
};

/**
 * Check if an image file is valid
 *
 * @param {File} file - File object
 * @returns {boolean} - Whether the file is a valid image
 */
export const isValidImageFile = (file) => {
  // Check if file exists
  if (!file) return false;

  // Check if it's an image file
  const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
  return validTypes.includes(file.type);
};

/**
 * Truncate text to a specific length
 *
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} - Truncated text
 */
export const truncateText = (text, maxLength = 100) => {
  if (!text) return '';
  if (text.length <= maxLength) return text;

  return text.substr(0, maxLength) + '...';
};

/**
 * Download data as a JSON file
 *
 * @param {Object} data - Data to download
 * @param {string} filename - Filename
 */
export const downloadJson = (data, filename = 'detection_results.json') => {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
};

/**
 * Sort detections by confidence score
 *
 * @param {Array} detections - Array of detection objects
 * @param {string} order - Sort order ('asc' or 'desc')
 * @returns {Array} - Sorted detections
 */
export const sortDetectionsByConfidence = (detections, order = 'desc') => {
  if (!detections || !Array.isArray(detections)) return [];

  return [...detections].sort((a, b) => {
    return order === 'desc'
      ? b.confidence - a.confidence
      : a.confidence - b.confidence;
  });
};

export default {
  formatFileSize,
  formatTime,
  getClassColor,
  isValidImageFile,
  truncateText,
  downloadJson,
  sortDetectionsByConfidence,
};