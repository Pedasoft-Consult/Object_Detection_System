# Setting Up Monitoring with Prometheus and Grafana

This guide will help you set up a comprehensive monitoring system for your Object Detection API using Prometheus and Grafana.

## Overview

The monitoring system consists of:

1. **Prometheus**: Collects and stores metrics from your API
2. **Grafana**: Visualizes the metrics in dashboards
3. **Node Exporter** (optional): Provides system metrics from your server

## 1. Installing Prometheus

### Prerequisites
- A Linux server (Ubuntu/Debian recommended)
- Docker (optional but recommended)

### Using Docker

```bash
# Create directories for Prometheus data
mkdir -p /path/to/prometheus/data

# Run Prometheus container
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v /path/to/prometheus/data:/prometheus \
  prom/prometheus
```

### Without Docker

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.42.0/prometheus-2.42.0.linux-amd64.tar.gz

# Extract the archive
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Copy the prometheus.yml file to the extracted directory
cp /path/to/prometheus.yml .

# Run Prometheus
./prometheus --config.file=prometheus.yml
```

## 2. Installing Grafana

### Using Docker

```bash
# Create directory for Grafana data
mkdir -p /path/to/grafana/data

# Run Grafana container
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -v /path/to/grafana/data:/var/lib/grafana \
  grafana/grafana
```

### Without Docker

```bash
# Add Grafana APT repository
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list

# Update package list and install Grafana
sudo apt-get update
sudo apt-get install grafana

# Start Grafana service
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

## 3. Node Exporter (Optional but Recommended)

To get system metrics (CPU, memory, disk, etc.), install Node Exporter:

### Using Docker

```bash
docker run -d \
  --name node-exporter \
  -p 9100:9100 \
  -v /proc:/host/proc:ro \
  -v /sys:/host/sys:ro \
  -v /:/rootfs:ro \
  --net="host" \
  prom/node-exporter \
  --path.procfs=/host/proc \
  --path.sysfs=/host/sys \
  --collector.filesystem.ignored-mount-points="^/(sys|proc|dev|host|etc)($$|/)"
```

### Without Docker

```bash
# Download Node Exporter
wget https://github.com/prometheus/node_exporter/releases/download/v1.5.0/node_exporter-1.5.0.linux-amd64.tar.gz

# Extract the archive
tar xvfz node_exporter-*.tar.gz
cd node_exporter-*

# Run Node Exporter
./node_exporter
```

## 4. Integrating with Your API

1. Copy the `api_monitoring.py` file to your API project directory
2. Import and use the monitoring functions in your FastAPI app:

```python
# In your main app.py file
from api_monitoring import setup_monitoring, track_inference_time, track_detections

# Initialize FastAPI app
app = FastAPI()

# Set up monitoring
setup_monitoring(app)

# Use the decorators in your endpoints
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Your existing code
    preprocessed_image, original_shape = preprocess_image(image)
    
    # Use the decorator to track inference time
    @track_inference_time
    def run_inference(image_data):
        outputs = model.session.run(None, {model.input_name: image_data})
        return outputs
    
    outputs = run_inference(preprocessed_image)
    
    # Process predictions
    detections = model.postprocess(outputs[0], original_shape)
    
    # Track detected objects
    track_detections(detections, model.class_names)
    
    return {"detections": detections, ...}
```

## 5. Configuring Prometheus

Make sure your `prometheus.yml` file contains the proper scrape configuration:

```yaml
scrape_configs:
  - job_name: 'object_detection_api'
    metrics_path: '/metrics'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8000']  # Update with your API's host:port
```

## 6. Setting Up Grafana Dashboard

1. Access Grafana at `http://your-server:3000`
2. Default login: admin/admin
3. Add Prometheus as a data source:
   - Go to Configuration > Data Sources > Add data source
   - Select Prometheus
   - URL: `http://localhost:9090` (or your Prometheus server address)
   - Click "Save & Test"
4. Import the dashboard:
   - Go to Create > Import
   - Paste the contents of `object_detection_dashboard.json` or upload the file
   - Select your Prometheus data source
   - Click "Import"

## 7. Metrics Available in Your Dashboard

Your dashboard will display:

- **API Response Time**: Time taken to process requests
- **Request Rate**: Number of requests per second
- **Average Inference Time**: Time the model takes to run inference
- **CPU Usage**: System CPU utilization
- **Memory Usage**: System memory utilization
- **Detections by Class**: Count of objects detected by class
- **Error Rate**: Rate of errors in your API

## 8. Creating Alerts (Optional)

In Grafana:

1. Edit a panel
2. Go to the "Alert" tab
3. Configure alert conditions (e.g., response time > 500ms)
4. Set notification channels (email, Slack, etc.)

## 9. Troubleshooting

### Prometheus Issues

- Check if Prometheus is running: `curl http://localhost:9090/-/healthy`
- Verify targets in Prometheus UI: `http://localhost:9090/targets`
- Check logs: `docker logs prometheus` (if using Docker)

### Grafana Issues

- Check if Grafana is running: `curl http://localhost:3000/api/health`
- Verify data source connection in Grafana UI
- Check logs: `docker logs grafana` (if using Docker)

### API Metrics Issues

- Make sure your API is exposing the `/metrics` endpoint
- Verify metrics with: `curl http://your-api:8000/metrics`
- Check for errors in your API logs

## Next Steps

Once your monitoring system is set up, consider:

1. Setting up **alerting** for critical conditions
2. Adding more **custom metrics** specific to your application
3. Creating **additional dashboards** for different aspects of your system
4. Setting up **long-term storage** with Prometheus remote write