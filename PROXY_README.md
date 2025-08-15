# Reverse Proxy for CORS Avoidance

This module provides a reverse proxy functionality to avoid CORS issues when connecting to external services like MuseTalk.

## Overview

The reverse proxy acts as an intermediary between the frontend and external services. Instead of the frontend making direct requests to external URLs (which can cause CORS issues), it makes requests to `/api/*` endpoints on the local server, which then forwards them to the target service.

## How It Works

1. **Frontend Request**: The frontend makes a request to `/api/{path}?target_url={encoded_url}`
2. **Proxy Processing**: The server extracts the target URL, strips the `/api` prefix from the path, and forwards the request
3. **Response**: The proxy returns the response from the target service

**Important**: The `/api` prefix is automatically stripped from the path when forwarding to the target server. For example:
- `/api/health` → `/health`
- `/api/stream` → `/stream`
- `/api/upload_answer` → `/upload_answer`

## Usage

### Basic Proxy Request

```javascript
// Instead of this (causes CORS):
fetch('http://external-service.com/api/endpoint')

// Use this:
fetch('/api/endpoint?target_url=http://external-service.com')
```

### With Headers

```javascript
// Using X-Target-URL header
fetch('/api/endpoint', {
  headers: {
    'X-Target-URL': 'http://external-service.com',
    'Content-Type': 'application/json'
  },
  method: 'POST',
  body: JSON.stringify(data)
})
```

### With Request Body

```javascript
// Include target_url in request body
fetch('/api/endpoint', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    target_url: 'http://external-service.com',
    data: { ... }
  })
})
```

## Available Endpoints

### Proxy Routes

- `GET /api/{path}` - Proxy GET requests
- `POST /api/{path}` - Proxy POST requests  
- `PUT /api/{path}` - Proxy PUT requests
- `DELETE /api/{path}` - Proxy DELETE requests
- `PATCH /api/{path}` - Proxy PATCH requests
- `OPTIONS /api/{path}` - Handle CORS preflight requests

### Utility Endpoints

- `GET /proxy/health` - Check proxy health
- `POST /proxy/check` - Test connection to target URL by checking its `/health` endpoint (returns success only for HTTP 200)

## Target URL Specification

The target URL can be provided in three ways:

1. **Query Parameter**: `?target_url=http://example.com`
2. **Header**: `X-Target-URL: http://example.com`
3. **Request Body**: `{"target_url": "http://example.com"}`

## Examples

### Testing Connection

```javascript
// Test if a service is reachable by checking its /health endpoint
const response = await fetch('/proxy/check', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: 'http://musetalk-server:7860' })
});

const result = await response.json();
if (result.ok) {
  console.log('✅ Health check successful:', result.message);
} else {
  console.log('❌ Health check failed:', result.message);
}
```

### Streaming Video

```javascript
// Stream MJPEG from MuseTalk
const streamUrl = 'http://musetalk-server:7860/stream';
const proxyUrl = `/api/stream?target_url=${encodeURIComponent(streamUrl)}`;

const response = await fetch(proxyUrl);
const reader = response.body.getReader();
// Process streaming data...
```

### File Upload

```javascript
// Upload file to external service
const formData = new FormData();
formData.append('file', file);
formData.append('target_url', 'http://external-service.com');

const response = await fetch('/api/upload', {
  method: 'POST',
  body: formData
});
```

## CORS Headers

The proxy automatically adds CORS headers to all responses:

- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, PUT, DELETE, PATCH, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, X-Target-URL`

## Error Handling

The proxy returns appropriate HTTP status codes:

- `400` - Missing or invalid target URL
- `500` - Proxy error
- `504` - Request timeout
- Original status codes from target service

## Testing

Run the test script to verify proxy functionality:

```bash
python test_proxy.py
```

## Integration with Existing Code

The proxy is automatically integrated into the Flask server. All existing functionality continues to work, but now external service calls go through the proxy to avoid CORS issues.

### Updated Functions

- `checkMusetalkConnection()` - Now uses `/proxy/check`
- `connectMusetalk()` - Now uses `/api/stream`
- `pollForStreamReady()` - Now uses `/api/stream_status`
- `_forward_answer_and_trigger_inference()` - Now uses `/api/upload_answer` and `/api/start`

## Security Considerations

- The proxy forwards all headers except `Host`, `Connection`, and `X-Target-URL`
- Target URLs are validated and normalized
- Request timeouts are enforced (default: 30 seconds)
- No authentication is performed - ensure your target services are properly secured
