#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'analiza_slik_edvarda_muncha' / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analiza_ene_slike as analyzer

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})


@app.route('/generate', methods=['GET', 'POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json() or {}
    filename = data.get('filename')
    print(f"[DEBUG] Received filename: {filename}")
    
    if not filename:
        return jsonify({'ok': False, 'error': 'missing filename'}), 400

    img_path = ROOT / 'munch_paintings' / filename
    print(f"[DEBUG] Looking for image at: {img_path}")
    print(f"[DEBUG] Image exists: {img_path.exists()}")
    
    if not img_path.exists():
        # List available files for debugging
        munch_dir = ROOT / 'munch_paintings'
        available = list(munch_dir.glob('*')) if munch_dir.exists() else []
        print(f"[DEBUG] Available files: {[f.name for f in available[:5]]}")
        return jsonify({'ok': False, 'error': f'source image not found: {filename}'}), 404

    try:
        analysis = analyzer.analyse_painting(str(img_path))
        analyzer.save_analysis_images(analysis)
        
        # Check where images were actually saved
        title = os.path.splitext(os.path.basename(str(img_path)))[0]
        expected_dir = ROOT / 'analiza_slik_edvarda_muncha' / 'web' / 'public' / 'generirani_grafi' / title
        print(f"[DEBUG] Expected output dir: {expected_dir}")
        print(f"[DEBUG] Dir exists: {expected_dir.exists()}")
        if expected_dir.exists():
            files = list(expected_dir.glob('*.png'))
            print(f"[DEBUG] PNG files in dir: {[f.name for f in files]}")
        
        return jsonify({'ok': True, 'generated_for': filename})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
