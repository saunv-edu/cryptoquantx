# web/dashboard.py
from flask import Flask, render_template, jsonify
import asyncio
import websockets
import json

class TradingDashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.live_data = {}
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/positions')
        def get_positions():
            return jsonify(self.get_current_positions())
        
        @self.app.route('/api/performance')
        def get_performance():
            return jsonify(self.get_performance_metrics())
        
        @self.app.route('/api/live_data')
        def get_live_data():
            return jsonify(self.live_data)