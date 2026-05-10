import random
import requests

class AttackGenerator:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url

    def generate_adaptive_adversarial(self, round_num, shap_feedback=None):
        evasive_templates = [
            "Hi {name}, I wanted to follow up on the document I shared last week. Could you take a quick look and let me know if everything looks good? Here's the link: {link}. Happy to discuss any questions. Best, {sender}",
            "{name}, The team has finalized the budget proposal. I've uploaded the latest version for your review — you can find it at {link}. No rush, but it would be great to have your sign-off by Friday. Cheers, {sender}",
            "Good afternoon {name}, Just a heads up — we're updating our internal tools next week. Please complete the setup at {link} when you get a moment. Let me know if you need any help. Thanks, {sender}",
            "Hi {name}, I came across an article I thought you'd find interesting for the {project} initiative. Check it out here: {link}. Would love to hear your perspective. - {sender}",
            "Hey {name}, Quick note — {sender_title} asked me to share the updated guidelines with you. They're available at {link}. Feel free to reach out if anything is unclear. Best regards, {sender}",
        ]

        if round_num >= 2 and shap_feedback:
            top_features = [f['feature'] for f in shap_feedback[:5]]
            if 'urgency_word_count' in top_features:
                evasive_templates = [t.replace("Please", "Feel free to").replace("need", "might want") for t in evasive_templates]
            if 'url_count' in top_features:
                evasive_templates.append(
                    "Hi {name}, I mentioned the {project} resource in our last call. "
                    "Here's what I was referring to — {link} — let me know your thoughts "
                    "when you get a chance. No pressure! {sender}")

        if round_num >= 3:
            evasive_templates += [
                "Hi {name}, Great catching up with you yesterday. As discussed, "
                "I've put together the materials for the {project} review. "
                "You can access everything through our shared workspace — "
                "the link is {link}. Looking forward to your feedback. {sender}",
                "{name}, Hope your week is going well. I wanted to share the "
                "competitive analysis our team completed. It's at {link}. "
                "Some interesting insights in there. Let's chat when you've "
                "had a chance to review. Best, {sender}",
            ]

        names = ["Sarah", "Michael", "Emily", "James", "Lisa", "David", "Jennifer", "Robert"]
        senders = ["Alex Rivera", "Priya Sharma", "Marcus Johnson", "Emma Chen"]
        projects = ["Horizon", "Catalyst", "Meridian", "Atlas"]
        links = ["https://workspace.company.com/shared/doc-" + str(random.randint(10000,99999)),
                 "https://portal.internal-tools.net/review/" + str(random.randint(10000,99999))]

        emails = []
        for _ in range(100):
            t = random.choice(evasive_templates)
            email = t.format(name=random.choice(names), sender=random.choice(senders),
                            project=random.choice(projects), link=random.choice(links),
                            sender_title=random.choice(["Director","VP","Team Lead"]))
            emails.append(email)
        return emails

    def attack_api(self, emails):
        results = []
        for i, email in enumerate(emails):
            try:
                resp = requests.post(f"{self.api_url}/predict", json={"email_text": email}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    results.append({
                        'email': email, 'classification': data['classification'],
                        'confidence': data['confidence_score'],
                        'shap_features': data['shap_features'],
                        'evaded': data['classification'] == 'legitimate'
                    })
            except Exception as e:
                print(f"Request {i} failed: {e}")
        return results

    def analyze_attack_results(self, results, round_num):
        total = len(results)
        evaded = sum(1 for r in results if r['evaded'])
        detected = total - evaded
        evasion_rate = evaded / max(total, 1)

        print(f"\\n--- Round {round_num} Attack Results ---")
        print(f"Total adversarial emails: {total}")
        print(f"Detected as phishing: {detected} ({detected/total*100:.1f}%)")
        print(f"Evaded detection: {evaded} ({evasion_rate*100:.1f}%)")
        print(f"Evasion Rate: {evasion_rate*100:.1f}%")

        if results:
            all_shap = []
            for r in results:
                if r['evaded'] and r.get('shap_features'):
                    for f in r['shap_features']:
                        all_shap.append(f['feature'])
            if all_shap:
                from collections import Counter
                feat_counts = Counter(all_shap).most_common(10)
                print(f"\\nTop exploited features (in evaded emails):")
                for feat, count in feat_counts:
                    print(f"  {feat}: appeared {count} times")

        return {'round': round_num, 'total': total, 'evaded': evaded,
                'detected': detected, 'evasion_rate': evasion_rate}
