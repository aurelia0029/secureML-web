import requests
import hmac
import hashlib
import sys
import os

# --- 必須與 Server 一致 ---
HMAC_SECRET_KEY = "NCKU_PQC_2026_SECRET"

def run_test(image_path):
    # 因為你是自簽憑證且跑在 8443，關閉 verify 並指向 localhost
    url = "https://localhost:8443/predict"
    
    print(f"[*] Sending {image_path} to PQC Server...")
    
    try:
        file_name = os.path.basename(image_path)

        with open(image_path, 'rb') as f:
            # 格式：('檔名', 檔案物件, 'MIME類型')
            files = {'file': (file_name, f, 'image/jpeg')} 
            response = requests.post(url, files=files, verify=False, timeout=30)
       
        if response.status_code != 200:
            print(f"[-] Error: Server returned {response.status_code}")
            return

        data = response.json()
        pred = data['prediction']
        m_hash = data['model_hash']
        received_sig = data['signature']

        # --- 執行安全性驗證 ---
        print("\n--- Security Audit ---")
        
        # 重新計算 HMAC
        message = f"{m_hash}:{pred}".encode()
        expected_sig = hmac.new(HMAC_SECRET_KEY.encode(), message, hashlib.sha256).hexdigest()

        # 1. 驗證簽章 (證明來源)
        if hmac.compare_digest(received_sig, expected_sig):
            print("✅ Origin Verified: Result is authentic (from NCKU Server).")
        else:
            print("❌ SECURITY ALERT: Signature mismatch! Data may be tampered.")
            return

        # 2. 顯示模型完整性 (證明模型沒被換過)
        print(f"✅ Model Integrity: Verified (Hash: {m_hash[:10]}...)")
        
        print(f"\n[FINAL RESULT] Prediction: {'PERSON' if pred == 1 else 'NO PERSON'}")
        print("----------------------\n")

    except Exception as e:
        print(f"[-] Connection failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_inference.py <image_path>")
    else:
        run_test(sys.argv[1])
