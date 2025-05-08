import os
import time
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# --- Load API key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment.")
print("🔑 API Key loaded.")
client = OpenAI(api_key=api_key)

# --- Test connection ---
print("🧪 Running test API call...")
try:
    test = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10
    )
    print("✅ Test response:", test.choices[0].message.content)
except Exception as e:
    print(f"❌ Test call failed:\n\n{e}\n")
    exit(1)

# --- Configuration ---
model = "gpt-4o"
total_conversations = 2000
output_file = "diverse_se_conversations_2000.csv"

# --- Prompt Components ---
recruiter_personas = [
    "an overly friendly recruiter trying to build quick rapport",
    "a formal, HR-style recruiter using corporate language",
    "a vague but persuasive executive recruiter",
    "a recruiter acting with urgency and pressure",
    "a recruiter claiming to be part of a stealth VC firm"
]

target_personas = [
    "a junior developer eager to impress",
    "a cautious and experienced cybersecurity analyst",
    "a busy senior engineer who gives short answers",
    "a skeptical DevOps manager who suspects phishing",
    "a compliance officer who rarely responds to recruiters"
]

bait_types = [
    "an urgent DevSecOps role with a $180K salary",
    "a remote classified project for a government partner",
    "an invitation-only executive track at a stealth startup",
    "an equity-funded AI lab that requires NDA access",
    "a fast-track cybersecurity grant that requires quick screening"
]

system_prompt = """
You are a synthetic data generator simulating realistic social engineering conversations on LinkedIn.
Each conversation must include:
- A fake recruiter trying to extract sensitive information
- A target employee who may or may not fall for the attempt

Formatting Rules:
- Generate EXACTLY 5 separate conversations per response
- Begin each with: **Conversation X** (X = 1–5)
- End each with a single label on a new line: [successful], [partial], or [failure]
- Use realistic LinkedIn-style tone and natural message flow
- Each conversation must contain 6–10 lines (3–5 messages per person)
- No summaries or extra commentary
"""

# --- Storage ---
conversations = []
labels = []

# --- Main Loop ---
pbar = tqdm(total=total_conversations)
while len(conversations) < total_conversations:
    try:
        recruiter = random.choice(recruiter_personas)
        target = random.choice(target_personas)
        bait = random.choice(bait_types)

        user_prompt = f"""
The recruiter is {recruiter}.
The target is {target}.
The bait is {bait}.

Generate 5 synthetic chat conversations between them using natural LinkedIn tone and varied interaction flow. Randomly vary the outcomes:
- [successful] — target shares sensitive info.
- [partial] — target shares something minor.
- [failure] — target refuses to engage or share anything.
"""

        print(f"⏰ Requesting at {datetime.now().strftime('%H:%M:%S')}... Total so far: {len(conversations)}")

        response = client.chat.completions.create(
            model=model,
            temperature=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        raw_output = response.choices[0].message.content
        print("📦 RAW RESPONSE:\n", raw_output[:300], "...\n---")

        chunks = raw_output.split("**Conversation ")
        for chunk in chunks:
            if not chunk.strip():
                continue

            chunk = "**Conversation " + chunk.strip()
            lines = chunk.strip().split("\n")
            label_line = lines[-1].strip().lower()

            if label_line.startswith("[") and label_line.endswith("]") and len(lines) >= 6:
                label = label_line.replace("[", "").replace("]", "")
                convo = "\n".join(lines[:-1]).strip()
                conversations.append(convo)
                labels.append(label)
                print(f"✅ Saved #{len(conversations)} — Label: {label}")
                pbar.update(1)

                if len(conversations) >= total_conversations:
                    break
            else:
                print("⚠️ Skipped: Improper format or too short.")

    except OpenAIError as e:
        print(f"❌ API Error: {e} — retrying in 10s...")
        time.sleep(10)
    except Exception as e:
        print(f"❌ General Error: {e} — retrying in 10s...")
        time.sleep(10)

# --- Save Output ---
df = pd.DataFrame({"conversation": conversations, "label": labels})
df.to_csv(output_file, index=False)
pbar.close()
print(f"\n✅ Saved {len(conversations)} conversations to: {output_file}")