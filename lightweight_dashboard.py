#!/usr/bin/env python3
"""
Lightweight Web Dashboard for Smart Contract AI Analyzer
Uses Flask with minimal HTML/CSS/JavaScript for fast, efficient UI
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import json
from datetime import datetime
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
API_BASE_URL = 'http://localhost:5000'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== Routes ====================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_contract():
    """Proxy endpoint for contract analysis"""
    try:
        data = request.get_json()
        contract_code = data.get('contract_code', '')
        
        if not contract_code.strip():
            return jsonify({'success': False, 'error': 'Contract code cannot be empty'}), 400
        
        # Call backend API
        response = requests.post(
            f'{API_BASE_URL}/api/analyze',
            json={'contract_code': contract_code},
            timeout=30
        )
        
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Analysis timeout - contract too large'}), 504
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check backend API health"""
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        return jsonify(response.json()), response.status_code
    except:
        return jsonify({'status': 'offline', 'error': 'Backend API unavailable'}), 503

@app.route('/api/models/status', methods=['GET'])
def models_status():
    """Get AI models status"""
    try:
        response = requests.get(f'{API_BASE_URL}/api/models/status', timeout=5)
        return jsonify(response.json()), response.status_code
    except:
        return jsonify({'error': 'Unable to fetch model status'}), 503

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.sol'):
            return jsonify({'success': False, 'error': 'Only .sol files allowed'}), 400
        
        # Read file content
        content = file.read().decode('utf-8')
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'content': content
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

# ==================== Main ====================

if __name__ == '__main__':
    print("🚀 Lightweight Dashboard Starting...")
    print("📊 Dashboard: http://localhost:8000")
    print("🔗 Backend API: http://localhost:5000")
    print("⚡ Lightweight UI - Minimal Resource Usage")
    print("-" * 50)
    
    app.run(host='0.0.0.0', port=8000, debug=False)
