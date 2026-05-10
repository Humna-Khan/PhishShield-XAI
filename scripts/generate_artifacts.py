import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Add src to path for internal modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from core.utils import extract_features

def main():
    print("Loading datasets...")
    
    # 1. Load Adversarial Samples (Phishing)
    adv_path = "data/adversarial_samples.csv"
    if os.path.exists(adv_path):
        adv_df = pd.read_csv(adv_path)
    else:
        print(f"Error: {adv_path} not found.")
        return

    # 2. Define a diverse set of Legitimate Samples
    # We add more links, technical terms (security, AI), and professional dates 
    # to teach the model that these aren't always malicious.
    legit_emails = [
        "Hi Team, just a reminder that our weekly sync is moved to 3 PM today. See you there!",
        "The quarterly report is attached. Please review the financial sections before the meeting.",
        "Good morning, I've updated the project roadmap on the shared drive. Let me know if you have questions.",
        "Hey, are we still on for lunch? I'm thinking of that new place downtown.",
        "Hi Sarah, great job on the presentation yesterday. The client was very impressed.",
        "Please find the meeting notes from this morning's session. Action items are highlighted.",
        "The office will be closed on Monday for the holiday. Enjoy the long weekend!",
        "Thanks for the help with the server migration. Everything seems to be running smoothly now.",
        "Hi Team, the Zoom link for our monthly all-hands meeting is: https://zoom.us/j/982374165. Please join early.",
        "Hi Sarah, I've posted the new 2026 benefits guide on our internal HR portal here: https://hr.internal-company.net/benefits-guide",
        "Hey, I came across this interesting article about AI trends: https://hbr.org/2026/05/ai-security-trends. Thought you'd like it!",
        "The security audit for Q2 is finalized. You can access the results at https://portal.internal.com/security/audit-results",
        "Regarding the Meridian project, here is the latest documentation: https://docs.google.com/document/d/project-roadmap-2026",
        "Hi, I've shared the design assets for the Atlas project on Figma: https://www.figma.com/file/atlas-design-system",
        "Please check the updated guidelines on our Wiki: https://wiki.company.org/engineering/coding-standards",
        "Here is the link to the recording of yesterday's training session: https://microsoftstream.com/video/training-101",
        "I've updated my LinkedIn profile with the new certification! Check it out here: https://www.linkedin.com/in/my-profile",
        "The reservation for the team dinner is confirmed at 7 PM. Map: https://maps.google.com/?q=restaurant-location",
        "The password for your corporate account was successfully changed on May 5, 2026. If you did not perform this action, please contact the security desk immediately. No further action is required.",
        "Your login to the VPN was successful from a new device. If this was you, no action is needed. Otherwise, contact IT support.",
        "Notice: Our corporate security policy has been updated. You can read the full documentation on the internal HR portal.",
        "Hi Team, the Zoom link for our all-hands meeting is https://zoom.us/j/123456789. Please join 5 minutes early.",
        "Here is the Teams meeting link for the roadmap discussion: https://teams.microsoft.com/l/meetup-join/roadmap-sync",
        "Hi Sarah, let's catch up on Google Meet: https://meet.google.com/abc-defg-hij. Talk soon!",
        "The password for your corporate account was successfully changed on May 5, 2026. If you did not perform this action, please contact the security desk immediately. No further action is required.",
        "Hi Sarah, per our discussion, here is the roadmap (https://docs.google.com/roadmap), the budget (https://sheets.google.com/budget), and the calendar (https://calendar.google.com/sync). Let me know if you need more info."
    ] * 100 # Stronger baseline
    
    legit_df = pd.DataFrame({
        'text': legit_emails,
        'label': 0,
        'is_adversarial': False
    })
    
    # 3. Adversarial Hardening: Add "Stealthy" Phishing patterns
    # These avoid urgency but still have malicious intent.
    stealthy_phishing = [
        "Hello, I'm following up on the invoice #83241 from last month. It seems the payment hasn't been processed yet. Review here: https://portal-billing-services.net/invoice/83241",
        "Hi Michael, As part of our annual security compliance audit, all employees need to re-verify their credentials: https://compliance-services-verify.org/portal",
        "Hi Sarah, I've posted the new 2026 benefits guide on our internal HR portal. Please verify your details: https://hr-portal-docs.com/verify-payroll",
        "Please review the updated guidelines for the Meridian project at the following link: https://secure-docs-access.com/r/928374",
        "Following up on our contract renewal — the revised invoice is ready for your review: https://portal-company-services.com/r/949360",
        "Your account requires a routine security check to maintain compliance with new regulations: https://security-verify-access.net/compliance",
        "URGENT: Your account access has been restricted. Please verify your details immediately: https://bit.ly/account-verify-2026",
        "Hi, I've shared the confidential salary spreadsheet with you. Access it here: https://tinyurl.com/salary-docs-restricted",
        "Final Reminder: Your subscription will be cancelled today. Update your payment info: https://bit.ly/pay-verify-msft",
        "Please review the updated HR policy regarding bonuses: https://tinyurl.com/hr-policy-update-2026",
        "Your package is waiting for delivery. Confirm your address here: https://bit.ly/package-delivery-verify",
        "Hi John, I'm tied up in meetings all day. Can you please send me a list of all active client wire transfer details? I need to verify them for the audit.",
        "URGENT: I need you to purchase 10 Apple Gift Cards for the team rewards program. Please send the codes to me once you have them.",
        "Hey, check out this private document: https://bit.ly/3xJ8k2z. It contains the salary spreadsheet.",
        "Please sign in to your Microsoft account to verify your subscription: https://mircosoft-verify.com/login",
        "Your account security is our priority. Verify here: https://googIe-security.net/r/8321"
    ] * 100 # Robust adversarial samples
    
    stealthy_df = pd.DataFrame({
        'text': stealthy_phishing,
        'label': 1,
        'is_adversarial': True
    })
    
    # Combine all
    df = pd.concat([adv_df, legit_df, stealthy_df]).reset_index(drop=True)
    print(f"Total samples: {len(df)} ({len(adv_df)+len(stealthy_df)} phishing, {len(legit_df)} legitimate)")

    # 3. Feature Extraction
    print("Extracting features...")
    
    # Linguistic Features (using the API's logic)
    ling_features_list = [extract_features(t) for t in df['text']]
    ling_df = pd.DataFrame(ling_features_list)
    ling_matrix = ling_df.values
    ling_names = list(ling_df.columns)
    
    # TF-IDF Features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text']).toarray()
    tfidf_names = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    
    # Combine Matrices
    X = np.hstack([ling_matrix, tfidf_matrix])
    y = df['label'].values
    feature_names = ling_names + tfidf_names
    
    # 4. Train Model with increased complexity
    print("Training Ensemble Model...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
    ensemble = VotingClassifier([('rf', rf), ('xgb', xgb)], voting='soft')
    ensemble.fit(X, y)
    
    # 5. Save Artifacts
    model_dir = "models/classical_model"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Saving artifacts to {model_dir}...")
    joblib.dump(ensemble, os.path.join(model_dir, "classical_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
    
    print("Success! All artifacts generated.")

if __name__ == "__main__":
    main()
