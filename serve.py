#!/usr/bin/env python3
"""
Simple HTTP server to serve the Bank Marketing ML Analytics Dashboard.
"""
import http.server
import socketserver
import os
import webbrowser

PORT = int(os.environ.get("PORT", 8080))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"📊 Bank Marketing ML Analytics Dashboard running at http://localhost:{PORT}")
        print(f"   Opening index.html and presentation.html...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()
