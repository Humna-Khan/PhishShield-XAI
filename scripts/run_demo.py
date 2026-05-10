import sys
import os
import time

# Add src to path for internal modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from simulator.attack_generator import AttackGenerator

def main():
    print("--- Phishing Detection Project Demo ---")
    print("Connecting to API at http://localhost:8000...")
    
    attacker = AttackGenerator(api_url="http://localhost:8000")
    
    # Wait for API to be ready
    import requests
    max_retries = 5
    for i in range(max_retries):
        try:
            resp = requests.get("http://localhost:8000/health")
            if resp.status_code == 200:
                print("API is healthy and ready.")
                break
        except:
            print(f"Waiting for API... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("API not reachable. Please ensure it's running.")
        return

    # Round 1: Standard Adversarial Attack
    print("\n--- Round 1: Baseline Adversarial Attack ---")
    emails = attacker.generate_adaptive_adversarial(round_num=1)
    results = attacker.attack_api(emails[:10]) # Test with a subset for speed
    attacker.analyze_attack_results(results, round_num=1)

    # Round 2: Adaptive Attack (simulating feedback loop)
    print("\n--- Round 2: Adaptive Attack (mimicking XAI-informed evasion) ---")
    # Get feedback from the first successful evasion
    feedback = results[0]['shap_features'] if results else None
    emails_v2 = attacker.generate_adaptive_adversarial(round_num=2, shap_feedback=feedback)
    results_v2 = attacker.attack_api(emails_v2[:10])
    attacker.analyze_attack_results(results_v2, round_num=2)

    print("\nProject is running successfully!")

if __name__ == "__main__":
    main()
