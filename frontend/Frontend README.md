# Object Detection System Frontend

This is the frontend application for the Object Detection System. It provides a user-friendly interface for uploading images, detecting objects, and viewing detection results.

## Project Structure

```
frontend/
├── public/               # Static files
├── src/
│   ├── components/       # React components
│   │   ├── Home.js       # Home page component
│   │   └── DetectionResults.js  # Results display component
│   ├── services/         # API client services
│   │   └── detectionApi.js  # Methods for communicating with the API
│   ├── utils/            # Utility functions
│   │   └── helpers.js    # Helper utilities
│   ├── App.css           # Main application styles
│   ├── App.js            # Main application component
│   ├── index.css         # Base styles
│   ├── index.js          # Application entry point
│   └── reportWebVitals.js  # Performance measurement
├── .env                  # Environment variables
├── package.json          # Dependencies and scripts
└── README.md             # Documentation
```

## Getting Started

### Prerequisites

- Node.js 14.x or higher
- npm 6.x or higher

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
Edit the `.env` file to set your API endpoint:
```
REACT_APP_API_URL=http://localhost:8000
```

### Development

Run the development server:
```bash
npm start
```

This will start the application in development mode at [http://localhost:3000](http://localhost:3000).

### Building for Production

Build the application for production:
```bash
npm run build
```

This creates optimized files in the `build` folder that can be deployed to a web server.

## Usage

1. Open the application in a web browser
2. Upload an image using the file selector
3. Adjust the confidence threshold and maximum detections if needed
4. Click "Detect Objects" to process the image
5. View the detection results with bounding boxes and class information

## Key Features

- Image upload and preview
- Configurable detection parameters
- Real-time results display
- Responsive design for desktop and mobile
- Detailed statistics and visualization

## Integration with Backend

This frontend communicates with the object detection API through the service defined in `src/services/detectionApi.js`. Ensure that your backend API is running and accessible at the URL specified in your `.env` file.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License.