# ============================================
# test_xaifa_api.py
# Test script to verify XAIFA API functionality
# ============================================

import requests
import json
import os

# Configuration
BASE_URL = "http://127.0.0.1:8001"
ASSETS_DIR = r"c:\Users\Aditya\OneDrive\Documents\New project\XAIFA\test_assets"

# Test assets
MODEL_FILE = os.path.join(ASSETS_DIR, "models", "tiny_rgb_classifier.pt")
DATASET_ZIP = os.path.join(ASSETS_DIR, "datasets", "folder_dataset_cat_dog.zip")
LABELS = "0: cat\n1: dog"


def print_response(label, response):
    """Print API response in formatted way."""
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    if hasattr(response, 'status_code'):
        print(f"Status: {response.status_code}")
        if response.status_code >= 400:
            print(f"Error: {response.text}")
            return None
    try:
        data = response.json() if hasattr(response, 'json') else response
        print(json.dumps(data, indent=2))
        return data
    except:
        print(response)
        return None


def main():
    print("\n" + "="*60)
    print("XAIFA API TEST SCRIPT")
    print("="*60)

    # Step 1: Health Check
    print("\n[1] Testing Health Endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/api/health")
        print_response("Health Check", resp)
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Step 2: Get Supported Models
    print("\n[2] Getting Supported Models...")
    try:
        resp = requests.get(f"{BASE_URL}/api/models/supported")
        print_response("Supported Models", resp)
    except Exception as e:
        print(f"Failed: {e}")

    # Step 3: Upload Model (tiny_rgb_classifier.pt)
    print("\n[3] Uploading Model (tiny_rgb_classifier.pt)...")
    model_id = None
    try:
        with open(MODEL_FILE, 'rb') as f:
            files = {'model_file': ('tiny_rgb_classifier.pt', f, 'application/octet-stream')}
            data = {
                'model_format': 'torchscript',
                'input_width': '64',
                'input_height': '64',
                'channels': '3',
                'num_classes': '2'
            }
            resp = requests.post(f"{BASE_URL}/api/models/upload", files=files, data=data)
        model_result = print_response("Model Upload", resp)
        model_id = model_result.get('model_id') if model_result else None
    except Exception as e:
        print(f"Model upload failed: {e}")

    # Try alternative model if first fails
    if not model_id:
        print("\n⚠️ Trying alternative model (simple_cnn_mnist_like.pth)...")
        ALT_MODEL = os.path.join(ASSETS_DIR, "models", "simple_cnn_mnist_like.pth")
        try:
            with open(ALT_MODEL, 'rb') as f:
                files = {'model_file': ('simple_cnn_mnist_like.pth', f, 'application/octet-stream')}
                data = {
                    'model_format': 'pytorch',
                    'architecture': 'simple_cnn',
                    'input_width': '28',
                    'input_height': '28',
                    'channels': '1',
                    'num_classes': '10'
                }
                resp = requests.post(f"{BASE_URL}/api/models/upload", files=files, data=data)
            model_result = print_response("Alternative Model Upload", resp)
            model_id = model_result.get('model_id') if model_result else None
        except Exception as e:
            print(f"Alternative model upload also failed: {e}")

    if not model_id:
        print("\n❌ Could not upload any model. Exiting.")
        return

    # Step 4: Upload Dataset
    print("\n[4] Uploading Dataset...")
    dataset_id = None
    try:
        with open(DATASET_ZIP, 'rb') as f:
            files = {'dataset_file': ('folder_dataset_cat_dog.zip', f, 'application/zip')}
            data = {
                'dataset_format': 'folder_zip',
                'labels': LABELS
            }
            resp = requests.post(f"{BASE_URL}/api/datasets/upload", files=files, data=data)
        dataset_result = print_response("Dataset Upload", resp)
        dataset_id = dataset_result.get('dataset_id') if dataset_result else None
    except Exception as e:
        print(f"Dataset upload failed: {e}")

    if not dataset_id:
        print("\n❌ Could not upload dataset. Exiting.")
        return

    # Step 5: Run Analysis
    print("\n[5] Running Analysis...")
    try:
        analysis_data = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "limit": 10
        }
        resp = requests.post(f"{BASE_URL}/api/runs/analyze", json=analysis_data)
        analysis_result = print_response("Analysis Results", resp)
        
        if analysis_result:
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"   Total Samples: {analysis_result.get('total_samples')}")
            print(f"   Correct: {analysis_result.get('correct_predictions')}")
            print(f"   Failed: {analysis_result.get('failed_predictions')}")
            print(f"   Accuracy: {analysis_result.get('accuracy', 0)*100:.1f}%")
            print(f"   Run ID: {analysis_result.get('run_id')}")
            
            failures = analysis_result.get('failures', [])
            if failures:
                print(f"\n⚠️ FAILURES DETECTED ({len(failures)} cases):")
                for f in failures[:5]:
                    print(f"   - True: {f.get('true_label')} | Predicted: {f.get('predicted_label')} | Conf: {f.get('confidence'):.2f}")
    except Exception as e:
        print(f"Analysis failed: {e}")

    # Step 6: Get Pipeline Info
    print("\n[6] Getting Pipeline Info...")
    try:
        resp = requests.get(f"{BASE_URL}/api/runs/pipeline")
        print_response("Pipeline Steps", resp)
    except Exception as e:
        print(f"Failed: {e}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()