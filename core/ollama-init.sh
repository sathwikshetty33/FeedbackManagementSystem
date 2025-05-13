#!/bin/bash

# This script helps set up and troubleshoot Ollama for use with the feedback insights system

echo "====== Feedback Insights Ollama Setup ======"
echo

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "Please install Ollama first from https://ollama.com/download"
    exit 1
else
    echo "✅ Ollama is installed"
fi

# Check if Ollama service is running
if curl -s --connect-timeout 5 http://localhost:11434/api/health &> /dev/null; then
    echo "✅ Ollama service is running"
else
    echo "❌ Ollama service is not running"
    echo "Starting Ollama service..."
    ollama serve &
    sleep 3
    
    # Check again
    if curl -s --connect-timeout 5 http://localhost:11434/api/health &> /dev/null; then
        echo "✅ Ollama service started successfully"
    else
        echo "❌ Failed to start Ollama service"
        echo "Please run 'ollama serve' manually"
        exit 1
    fi
fi

# Check if gemma:2b model is available
if ollama list | grep -q "gemma:2b"; then
    echo "✅ Model 'gemma:2b' is available"
else
    echo "❌ Model 'gemma:2b' is not available"
    echo "Pulling model 'gemma:2b'..."
    ollama pull gemma:2b
    
    # Check again
    if ollama list | grep -q "gemma:2b"; then
        echo "✅ Model 'gemma:2b' pulled successfully"
    else
        echo "❌ Failed to pull 'gemma:2b' model"
        exit 1
    fi
fi

# Test the model with a simple request
echo "Testing the model with a simple request..."
response=$(curl -s -X POST http://localhost:11434/api/generate -d '{
  "model": "gemma:2b",
  "prompt": "Say hello world",
  "stream": false
}')

if echo "$response" | grep -q "response"; then
    echo "✅ Model 'gemma:2b' is working correctly"
else
    echo "❌ Model test failed"
    echo "Response: $response"
    echo "Please check if Ollama is running properly"
    exit 1
fi

# Create .env file with Ollama settings
echo "Setting up environment variables..."
cat > .env << EOL
# Ollama API settings
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_TIMEOUT=60
MAX_OLLAMA_WORKERS=2
INSIGHT_CHUNK_SIZE=10
MAX_TOKENS_CHUNK=800
MAX_TOKENS_SYNTHESIS=1500
EOL

echo "✅ Created .env file with Ollama settings"
echo

echo "====== Setup Complete ======"
echo "You can now run your feedback insights system."
echo "To test the API, use: curl -X POST http://localhost:8000/insights/generate-insights/ -H 'Content-Type: application/json' -d '{\"event_id\": \"your_event_id\"}'"
echo
echo "Troubleshooting tips:"
echo "1. If the API returns 503, check if Ollama service is running with 'ollama serve'"
echo "2. To monitor Ollama logs, run 'tail -f ~/.ollama/logs/ollama.log'"
echo "3. If the model is running slowly, try decreasing MAX_OLLAMA_WORKERS in .env"
echo "4. For better performance, consider using a machine with more resources"