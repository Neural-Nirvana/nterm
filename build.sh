#!/bin/bash

# Build and publish script for alpa PyPI package

set -e  # Exit on any error

echo "🚀 Building and publishing alpa package..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade pip setuptools wheel twine build

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

# Build the package
echo "🏗️ Building package..."
python -m build

# Check the built package
echo "🔍 Checking package..."
twine check dist/*

# Upload to PyPI (uncomment when ready)
echo "📤 Ready to upload to PyPI!"
echo "To upload to TestPyPI: twine upload --repository testpypi dist/*"
echo "To upload to PyPI: twine upload dist/*"

# Optional: Upload to TestPyPI for testing
read -p "Upload to TestPyPI? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo "✅ Package uploaded to TestPyPI!"
    echo "Test install with: pip install --index-url https://test.pypi.org/simple/ alpa"
fi

echo "✅ Build process completed!"