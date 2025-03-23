# Swagger Documentation Guide

This guide explains how to add Swagger (OpenAPI) documentation to your Object Detection API. Swagger provides interactive API documentation that allows users to explore and test your API endpoints directly from a web browser.

## Overview

We've prepared the following components:

1. **OpenAPI Specification** (`openapi.json`) - A complete JSON document that describes your API endpoints, parameters, request/response formats, and more.
2. **Swagger UI Integration** (`swagger_ui.py`) - FastAPI integration code to set up the Swagger UI interface and serve your API documentation.

## Installation Steps

### 1. Add the OpenAPI specification file

Copy the `openapi.json` file to the root directory of your API project.

### 2. Add the Swagger UI integration

Copy the `swagger_ui.py` file to your project, typically in the same directory as your main app code or in a utils/helpers directory.

### 3. Integrate with your FastAPI app

Update your main app file (`app.py`) to use the Swagger UI integration:

```python
from fastapi import FastAPI
from swagger_ui import setup_swagger

# Create FastAPI app
app = FastAPI(
    # Disable the default docs
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Set up custom Swagger UI
setup_swagger(
    app,
    title="Object Detection API",
    description="API for real-time object detection using YOLO models",
    version="1.0.0"
)

# Rest of your FastAPI code...
```

### 4. Create static directory for Swagger UI assets

Create the following directory structure:

```
static/
├── openapi/
├── swagger-ui-bundle.js
├── swagger-ui.css
└── redoc.standalone.js
```

Download the required files:
- [swagger-ui-bundle.js](https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js)
- [swagger-ui.css](https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css)
- [redoc.standalone.js](https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js)

You can download these files using:

```bash
mkdir -p static/openapi

# Download Swagger UI assets
curl -o static/swagger-ui-bundle.js https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js
curl -o static/swagger-ui.css https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css
curl -o static/redoc.standalone.js https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js
```

## Customizing the API Documentation

### Modifying the OpenAPI Specification

You can update the `openapi.json` file to:

1. **Add new endpoints**: Add new entries to the `paths` object
2. **Update endpoint details**: Modify descriptions, parameters, request bodies, etc.
3. **Add models/schemas**: Add new schemas to the `components.schemas` object
4. **Update API metadata**: Update the title, description, version, etc.

Example of adding a new endpoint:

```json
"/custom_endpoint": {
  "post": {
    "summary": "Custom endpoint",
    "description": "Description of what this endpoint does",
    "operationId": "customEndpoint",
    "tags": ["custom"],
    "responses": {
      "200": {
        "description": "Successful response",
        "content": {
          "application/json": {
            "schema": {
              "$ref": "#/components/schemas/CustomResponse"
            }
          }
        }
      }
    }
  }
}
```

### Auto-generating from FastAPI

If you prefer to let FastAPI automatically generate the OpenAPI specification, you can modify the `setup_swagger` function to not load the custom file:

```python
def setup_swagger(app: FastAPI, title="Object Detection API", 
                description="API for real-time object detection using YOLO models",
                version="1.0.0"):
    # Mount static files directory
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Custom docs endpoints
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI"""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{title} - Swagger UI",
            oauth2_redirect_url="/docs/oauth2-redirect",
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )
    
    # ... rest of the function
```

## Accessing the Documentation

Once integrated, your API documentation will be available at:

- **Swagger UI**: `http://your-api-host:port/docs`
- **ReDoc**: `http://your-api-host:port/redoc`
- **Raw OpenAPI JSON**: `http://your-api-host:port/openapi.json`

## Securing the Documentation

For production environments, you might want to protect your API documentation. Here are some options:

1. **Basic authentication**: Add authentication middleware to protect the `/docs` and `/redoc` endpoints
2. **Disable in production**: Conditionally set up Swagger only in development environments
3. **OAuth2 security**: Configure OAuth2 security in your FastAPI app and Swagger UI

Example of conditionally enabling Swagger in development:

```python
import os

# Create FastAPI app
app = FastAPI()

# Set up Swagger only in development
if os.environ.get("ENVIRONMENT") != "production":
    setup_swagger(app)
```

## Troubleshooting

### Common Issues

1. **Swagger UI not loading**: Check that static files are correctly mounted and accessible
2. **OpenAPI specification not found**: Verify the path to your `openapi.json` file
3. **Endpoints missing from documentation**: Make sure they're correctly defined in the OpenAPI spec

### Checking for Errors

If you're having issues, check:

1. The FastAPI logs for any errors
2. The browser console for JavaScript errors
3. The network tab in your browser's developer tools to see if files are being loaded

## Further Customization

You can further customize your Swagger UI:

1. **Theme**: Modify the CSS or use a custom Swagger UI theme
2. **Branding**: Add your company logo and colors
3. **Plugins**: Add Swagger UI plugins for additional functionality

For more information, refer to:
- [OpenAPI Specification](https://swagger.io/specification/)
- [Swagger UI Documentation](https://swagger.io/tools/swagger-ui/)
- [FastAPI OpenAPI Documentation](https://fastapi.tiangolo.com/advanced/extending-openapi/)