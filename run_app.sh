#!/bin/bash

echo "========================================"
echo "    Fake News Detection AI"
echo "========================================"
echo ""
echo "Starting the application..."
echo ""
echo "The web interface will be available at:"
echo "http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    python3 app.py
elif command -v python &> /dev/null; then
    python app.py
else
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi
