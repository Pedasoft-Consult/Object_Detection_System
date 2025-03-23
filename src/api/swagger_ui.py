from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import json
import os


def setup_swagger(app: FastAPI, title="Object Detection API",
                  description="API for real-time object detection using YOLO models",
                  version="1.0.0"):
    """
    Set up custom Swagger UI for the FastAPI application

    Args:
        app: FastAPI application
        title: API title
        description: API description
        version: API version
    """

    # Create directory for OpenAPI file if it doesn't exist
    os.makedirs("static/openapi", exist_ok=True)

    # Try to load custom OpenAPI spec if exists
    custom_openapi_path = "openapi.json"
    if os.path.exists(custom_openapi_path):
        try:
            with open(custom_openapi_path, "r") as f:
                custom_openapi = json.load(f)

            # Make sure it has the openapi version field
            if "openapi" not in custom_openapi:
                custom_openapi["openapi"] = "3.0.0"

            # Save for static serving
            with open("static/openapi/openapi.json", "w") as f:
                json.dump(custom_openapi, f)

            # Define custom OpenAPI function to use the loaded spec
            def custom_openapi():
                return custom_openapi

            app.openapi = custom_openapi

        except Exception as e:
            print(f"Error loading custom OpenAPI spec: {e}")

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

    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc_html():
        """Custom ReDoc"""
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=f"{title} - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )

    @app.get("/openapi.json", include_in_schema=False)
    async def get_openapi_json():
        """Return OpenAPI schema as JSON"""
        if hasattr(app, "openapi_schema"):
            return JSONResponse(app.openapi_schema)

        # Generate OpenAPI schema from app routes
        openapi_schema = get_openapi(
            title=title,
            version=version,
            description=description,
            routes=app.routes,
        )

        # Explicitly set the OpenAPI version field
        if "openapi" not in openapi_schema:
            openapi_schema["openapi"] = "3.0.0"

        # Save for static serving
        with open("static/openapi/openapi.json", "w") as f:
            json.dump(openapi_schema, f)

        # Cache and return
        app.openapi_schema = openapi_schema
        return JSONResponse(app.openapi_schema)