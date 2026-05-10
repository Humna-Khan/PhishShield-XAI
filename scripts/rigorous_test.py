import requests
import json

api_url = "http://localhost:8000/predict"

test_cases = [
    # 1. Obvious Phishing
    {"text": "URGENT: Your account has been compromised. Click here to reset your password immediately: https://verify-identity-now.com/login", "expected": "phishing"},
    
    # 2. Stealthy Phishing (Adversarial)
    {"text": "Hello, I'm following up on the invoice #83241 from last month. It seems the payment hasn't been processed yet. Review here: https://portal-billing-services.net/invoice/83241", "expected": "phishing"},
    {"text": "Hi Michael, As part of our annual security compliance audit, all employees need to re-verify their access credentials: https://compliance-services-verify.org/portal", "expected": "phishing"},
    
    # 3. Legitimate Professional
    {"text": "Hi Team, the Zoom link for our all-hands meeting is https://zoom.us/j/123456789. Please join 5 minutes early.", "expected": "legitimate"},
    {"text": "Hi Sarah, I've posted the new 2026 benefits guide on our internal HR portal here: https://hr.company.com/benefits-2026", "expected": "legitimate"},
    {"text": "Hey, I came across this interesting article about AI trends: https://hbr.org/2026/05/ai-security-trends. Thought you'd like it!", "expected": "legitimate"},
    
    # 4. Legitimate Security/Transactional
    {"text": "The password for your corporate account was successfully changed on May 5, 2026. If you did not perform this action, contact the security desk.", "expected": "legitimate"},
    {"text": "Your login to the VPN was successful from a new device. No action is required if this was you.", "expected": "legitimate"},
    
    # 5. Casual Legitimate
    {"text": "Hey, are we still on for lunch? I'm thinking of that new place downtown.", "expected": "legitimate"},
    {"text": "Great job on the presentation! The client was really impressed with the roadmap.", "expected": "legitimate"},
    
    # 6. Extreme Edge Cases (Lookalike Domains & Homographs)
    {"text": "Please sign in to your Microsoft account to verify your subscription: https://mircosoft-verify.com/login", "expected": "phishing"},
    {"text": "Your account security is our priority. Verify here: https://googIe-security.net/r/8321", "expected": "phishing"}, # Capital 'I' in Google
    
    # 7. Shortened URLs & Redirects
    {"text": "Hey, check out this private document: https://bit.ly/3xJ8k2z. It contains the salary spreadsheet.", "expected": "phishing"},
    {"text": "The new project guidelines are available at https://tinyurl.com/project-docs-2026", "expected": "phishing"},
    
    # 8. Complex Professional (Multi-link)
    {"text": "Hi Sarah, per our discussion, here is the roadmap (https://docs.google.com/roadmap), the budget (https://sheets.google.com/budget), and the calendar (https://calendar.google.com/sync). Let me know if you need more info.", "expected": "legitimate"},
    
    # 9. Pure Text Phishing (Business Email Compromise)
    {"text": "Hi John, I'm tied up in meetings all day. Can you please send me a list of all active client wire transfer details? I need to verify them for the audit.", "expected": "phishing"}
]

def run_tests():
    print(f"{'Test Case':<70} | {'Expected':<10} | {'Actual':<10} | {'Status'}")
    print("-" * 110)
    
    all_passed = True
    for case in test_cases:
        try:
            resp = requests.post(api_url, json={"email_text": case["text"]})
            data = resp.json()
            actual = data["classification"]
            status = "✅ PASS" if actual == case["expected"] else "❌ FAIL"
            if actual != case["expected"]:
                all_passed = False
            print(f"{case['text'][:67] + '...':<70} | {case['expected']:<10} | {actual:<10} | {status}")
        except Exception as e:
            print(f"Error testing case: {e}")
            all_passed = False
            
    return all_passed

if __name__ == "__main__":
    if run_tests():
        print("\nAll tests passed! The model is now robust.")
    else:
        print("\nSome tests failed. Further refinement needed.")
