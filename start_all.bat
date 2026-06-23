@echo off
echo Starting Shopify AI Agent...
start "Image Generator" /MIN python "C:\Users\Iliyan\Desktop\image-generator\server.py"
timeout /t 3 /nobreak >nul
start "Shopify Agent" /MIN python "C:\shopify-agent\server.py"
timeout /t 2 /nobreak >nul
start "" "http://localhost:5001"
