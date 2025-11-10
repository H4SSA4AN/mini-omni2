import requests
import logging
from urllib.parse import urljoin, urlparse
import time
from flask import request, jsonify, Response, stream_with_context
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReverseProxy:
    def __init__(self, default_timeout=30):
        self.default_timeout = default_timeout
        self.session = requests.Session()
        
    def normalize_url(self, url):
        """Normalize URL to ensure it has a protocol"""
        if not url:
            return None
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        return url
    
    def proxy_request(self, path=''):
        """Handle proxy requests to /api/* endpoints"""
        try:
            # Extract the target URL from request headers, body, or query parameters
            target_url = None
            
            # Try to get URL from headers first
            target_url = request.headers.get('X-Target-URL')
            
            # If not in headers, try to get from request body for POST requests
            if not target_url and request.method == 'POST':
                try:
                    if request.is_json:
                        target_url = request.get_json().get('target_url')
                    else:
                        target_url = request.form.get('target_url')
                except:
                    pass
            
            # If still no URL, try to get from query parameters
            if not target_url:
                target_url = request.args.get('target_url')
            
            if not target_url:
                return jsonify({
                    'error': 'No target URL provided. Use X-Target-URL header, target_url in body, or target_url query parameter.'
                }), 400
            
            target_url = self.normalize_url(target_url)
            if not target_url:
                return jsonify({'error': 'Invalid target URL'}), 400
            
            # Strip /api prefix from path if present
            if path.startswith('api/'):
                path = path[4:]  # Remove 'api/' prefix
            elif path == 'api':
                path = ''  # If just 'api', make it empty
            
            # Construct the full target URL
            full_target_url = urljoin(target_url, path)
            
            # Add query string if present
            if request.query_string:
                separator = '&' if '?' in full_target_url else '?'
                full_target_url += separator + request.query_string.decode()
            
            logger.info(f"Proxying {request.method} {request.path} -> {full_target_url}")
            
            # Prepare headers (exclude host and connection headers)
            headers = dict(request.headers)
            headers.pop('Host', None)
            headers.pop('Connection', None)
            headers.pop('X-Target-URL', None)  # Remove our custom header
            
            # Prepare request data
            data = None
            json_data = None
            files = None
            
            if request.method in ['POST', 'PUT', 'PATCH']:
                if request.is_json:
                    json_data = request.get_json()
                elif request.files:
                    files = request.files
                elif request.form:
                    data = request.form
                else:
                    data = request.get_data()
            
            # Make the proxy request
            start_time = time.time()
            
            response = self.session.request(
                method=request.method,
                url=full_target_url,
                headers=headers,
                json=json_data,
                data=data,
                files=files,
                stream=True,  # Stream response for real-time data
                timeout=self.default_timeout,
                allow_redirects=False
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Proxy response: {response.status_code} ({elapsed_ms:.1f}ms)")
            
            # Handle streaming responses (like MJPEG or audio streaming)
            if response.headers.get('content-type', '').startswith(('multipart/', 'video/', 'audio/')):
                def generate():
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            yield chunk
                
                return Response(
                    stream_with_context(generate()),
                    status=response.status_code,
                    headers=dict(response.headers)
                )
            
            # Handle regular responses
            content = response.content
            return Response(
                content,
                status=response.status_code,
                headers=dict(response.headers)
            )
                
        except requests.exceptions.Timeout:
            logger.error("Proxy request timed out")
            return jsonify({'error': 'Request timed out'}), 504
        except Exception as e:
            logger.error(f"Proxy error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Proxy error: {str(e)}'}), 500
    
    def health_check(self):
        """Health check endpoint"""
        return jsonify({
            'status': 'ok',
            'service': 'reverse_proxy',
            'timestamp': time.time()
        })
    
    def check_target_connection(self):
        """Check connection to target URL by testing the /health endpoint"""
        try:
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
            
            target_url = self.normalize_url(data.get('url'))
            
            if not target_url:
                return jsonify({'ok': False, 'error': 'No URL provided'})
            
            # Normalize the base URL (remove any paths like /stream, /health, etc.)
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(target_url)
            base_url = urlunparse((parsed.scheme or 'http', parsed.netloc, '', '', '', ''))
            
            # Build our public base for MuseTalk to register stream callbacks
            xf_proto = request.headers.get('X-Forwarded-Proto')
            xf_host = request.headers.get('X-Forwarded-Host')
            scheme = xf_proto or request.scheme
            host = xf_host or request.host
            public_base = f"{scheme}://{host}"

            # Test the /health endpoint with stream_base param
            health_url = f"{base_url}/health?stream_base={public_base}"
            start_time = time.time()
            
            response = self.session.get(health_url, timeout=10)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Only return success if status is exactly 200
            is_ok = response.status_code == 200
            
            return jsonify({
                'ok': is_ok,
                'status': response.status_code,
                'url': health_url,
                'elapsed_ms': elapsed_ms,
                'message': 'Health check successful' if is_ok else f'Health check failed with status {response.status_code}'
            })
                
        except Exception as e:
            return jsonify({
                'ok': False,
                'error': str(e),
                'message': f'Connection error: {str(e)}'
            })

def create_proxy_routes(app, proxy_instance=None):
    """Add proxy routes to a Flask app"""
    if proxy_instance is None:
        proxy_instance = ReverseProxy()
    
    # Add CORS headers to all proxy responses
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, PATCH, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Target-URL'
        return response
    
    # Add proxy routes with CORS
    def proxy_with_cors(path):
        response = proxy_instance.proxy_request(path)
        return add_cors_headers(response)
    
    def health_with_cors():
        response = proxy_instance.health_check()
        return add_cors_headers(response)
    
    def check_with_cors():
        response = proxy_instance.check_target_connection()
        return add_cors_headers(response)
    
    app.add_url_rule('/api/<path:path>', 'proxy_api', 
                     proxy_with_cors, 
                     methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
    
    # Also handle /api without additional path
    app.add_url_rule('/api', 'proxy_api_root', 
                     lambda: proxy_with_cors(''), 
                     methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
    
    # Add health and check endpoints
    app.add_url_rule('/proxy/health', 'proxy_health', 
                     health_with_cors, methods=['GET'])
    app.add_url_rule('/proxy/check', 'proxy_check', 
                     check_with_cors, methods=['POST'])
    
    return proxy_instance
