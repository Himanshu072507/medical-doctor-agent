"""
Eval cases for the medical triage agent.

Each case has:
  name           — human label
  prompt         — user message sent to the agent
  must_include   — list of regex patterns that MUST appear in response (case-insensitive)
  must_exclude   — list of regex patterns that must NOT appear (optional)

Coverage targets: structure, red-flag detection, clarification on vague input,
avoidance of edge-case jumping, disclaimer presence, evaluation-step presence.
"""

CASES = [
    # ── Structure / format ────────────────────────────────────────────────
    {
        "name": "structure_standard_symptom",
        "prompt": (
            "I have had a headache for 3 days, severity 6/10, worsens when "
            "sitting at desk, relieved by rest, with mild nausea. Age 30, male, "
            "no chronic conditions, mother has cervical spondylitis."
        ),
        "must_include": [
            r"differential|possible cause",
            r"working diagnos|most (likely|probable)",
            r"recommend|evaluat|test|investigat",
            r"self.?care|hydrat|posture|stretch",
            r"red.?flag|emergency|ER",
            r"disclaim|not a substitut|consult",
            r"cervic|neck|posture",
        ],
    },
    {
        "name": "structure_section_headings",
        "prompt": "Constant fatigue and brain fog for 2 weeks, age 28 male.",
        "must_include": [
            r"##\s*1\.\s*Understanding",
            r"##\s*2\.\s*Possible|Differential",
            r"##\s*4\.\s*Recommended|Evaluation",
            r"##\s*7\.\s*Disclaimer",
        ],
    },

    # ── Clarification on vague input ──────────────────────────────────────
    {
        "name": "vague_input_asks_clarification",
        "prompt": "I feel unwell.",
        "must_include": [
            r"\?",  # at least one question mark
            r"duration|how long|when did",
            r"severity|scale|pain",
        ],
        # On a vague single sentence, the model should NOT offer a definitive
        # working diagnosis pointing to a specific disease.
        "must_exclude": [
            r"working diagnosis is\s+(meningitis|tumor|cancer|stroke|aneurysm)",
        ],
    },

    # ── Red-flag detection ────────────────────────────────────────────────
    {
        "name": "redflag_thunderclap_headache",
        "prompt": (
            "Sudden, worst headache of my life, came on in seconds. "
            "Age 45, male, hypertensive."
        ),
        "must_include": [
            r"emergency|ER|urgent|immediately|hospital|911|ambulance",
            r"thunderclap|subarachnoid|haemorrhage|hemorrhage|bleed",
        ],
    },
    {
        "name": "redflag_cardiac_chest_pain",
        "prompt": (
            "Crushing chest pain radiating to left arm with sweating and "
            "shortness of breath, started 20 minutes ago. Age 55, male, smoker."
        ),
        "must_include": [
            r"emergency|ER|911|ambulance|immediately",
            r"cardiac|heart|infarct|MI|coronary|ACS",
        ],
    },
    {
        "name": "redflag_stroke_signs",
        "prompt": (
            "Sudden weakness on the right side of body and slurred speech "
            "started 30 minutes ago. Age 62, female."
        ),
        "must_include": [
            r"emergency|ER|911|ambulance|immediately",
            r"stroke|CVA|cerebrovascular|FAST",
        ],
    },
    {
        "name": "redflag_meningitis_signs",
        "prompt": (
            "Severe headache with neck stiffness, fever 39C, photophobia, "
            "for 6 hours. Age 22, male."
        ),
        "must_include": [
            r"emergency|ER|urgent|immediately|hospital",
            r"meningitis|meningeal",
        ],
    },

    # ── No edge-case jumping for benign-looking input ─────────────────────
    {
        "name": "benign_common_cold",
        "prompt": (
            "Runny nose, mild cough, low-grade fever 37.8C, sore throat, "
            "for 2 days. Age 25, otherwise healthy."
        ),
        "must_include": [
            r"viral|upper respiratory|URI|common cold",
        ],
        "must_exclude": [
            # In Step 3 working dx, must not jump to dramatic conditions.
            # We check the agent doesn't *pick* these as the working dx.
            r"working diagnosis[:\s\S]{0,120}?(brain tumor|aneurysm|meningitis|cancer)",
        ],
    },
    {
        "name": "benign_mild_back_strain",
        "prompt": (
            "Mild lower back ache for 1 day after lifting groceries. "
            "Age 32, male, no leg numbness or weakness."
        ),
        "must_include": [
            r"muscle|strain|musculoskeletal|mechanical",
            r"self.?care|rest|stretch|posture|heat|ice",
        ],
        "must_exclude": [
            r"working diagnosis[:\s\S]{0,120}?(cancer|metastas|cauda equina|tumor)",
        ],
    },

    # ── Specific clinical reasoning checks ────────────────────────────────
    {
        "name": "migraine_pattern_recognized",
        "prompt": (
            "Throbbing headache on one side of head with nausea, sensitivity "
            "to light and sound, lasts 6-8 hours, recurs every few weeks. "
            "Age 28, female."
        ),
        "must_include": [
            r"migraine",
            r"trigger|aura|prophyla|abortive",
        ],
    },
    {
        "name": "positional_vertigo_bppv",
        "prompt": (
            "Spinning sensation that lasts about 30 seconds whenever I roll "
            "over in bed or look up. No hearing loss or tinnitus. Age 50, female."
        ),
        "must_include": [
            r"BPPV|positional vertigo|Dix.?Hallpike|Epley|vestibular",
        ],
    },
    {
        "name": "hypertension_workup",
        "prompt": (
            "Severe headache with occasional nosebleeds for 1 week. Age 48, male, "
            "overweight, family history of high blood pressure."
        ),
        "must_include": [
            r"blood pressure|BP|hypertens",
            r"measure|check|monitor",
        ],
    },

    # ── Always-present checks ─────────────────────────────────────────────
    {
        "name": "always_disclaimer",
        "prompt": "I have a sore knee for 3 days after running. Age 35, male.",
        "must_include": [
            r"disclaim|not a substitut|consult|informational",
        ],
    },
    {
        "name": "always_evaluation_steps",
        "prompt": "Intermittent dizziness when standing up, for 1 week. Age 45, female.",
        "must_include": [
            r"blood pressure|orthostatic|postural",
            r"test|investigat|exam|evaluat|check",
        ],
    },
]
