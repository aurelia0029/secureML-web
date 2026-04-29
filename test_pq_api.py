#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test client for People Detection API with Post-Quantum TLS
"""

import requests
import urllib3
from pathlib import Path

# Disable SSL warnings for self-signed certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = "https://localhost:8443"


def test_health():
    """Test health endpoint"""
    print("=" * 70)
    print("Testing Health Endpoint...")
    print("=" * 70)

    response = requests.get(f"{API_BASE}/health", verify=False)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_root():
    """Test root endpoint"""
    print("=" * 70)
    print("Testing Root Endpoint...")
    print("=" * 70)

    response = requests.get(f"{API_BASE}/", verify=False)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_predict(image_path: str):
    """Test prediction endpoint"""
    print("=" * 70)
    print("Testing Prediction Endpoint...")
    print("=" * 70)

    img_file = Path(image_path)
    if not img_file.exists():
        print(f"⚠ Image not found: {image_path}")
        print("Skipping prediction test...")
        return

    with open(img_file, "rb") as f:
        files = {"file": (img_file.name, f, "image/jpeg")}
        response = requests.post(f"{API_BASE}/predict", files=files, verify=False)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def check_tls_info():
    """Check TLS connection info"""
    print("=" * 70)
    print("TLS Connection Information")
    print("=" * 70)
    print("Note: To verify Post-Quantum key exchange, use:")
    print("  openssl s_client -connect localhost:8443 -servername localhost")
    print()
    print("Or capture packets with:")
    print("  sudo tcpdump -i lo0 -w api_pq_tls.pcap port 8443")
    print()
    print("Then analyze with:")
    print("  tshark -r api_pq_tls.pcap -Y \"tls.handshake.extensions_supported_groups\" -V")
    print("=" * 70)
    print()


if __name__ == "__main__":
    print("🔒 People Detection API - Post-Quantum TLS Test Client")
    print()

    try:
        # Test basic endpoints
        test_root()
        test_health()

        # Test prediction with a sample image
        # You can change this path to any image file
        sample_image = "inference_img/shanghai_test_1.png"
        test_predict(sample_image)

        # Show TLS info
        check_tls_info()

        print("✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("Make sure the server is running:")
        print("  python app_https.py")
    except Exception as e:
        print(f"❌ Error: {e}")
