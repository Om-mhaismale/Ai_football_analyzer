from transformers import pipeline

# Load the model
commentator = pipeline(
    "text-generation",
    model="distilgpt2",
    device_map="auto"
)

last_commentary = ""

def generate_commentary(positions):
    global last_commentary

    situation = []
    if positions.get("ball"):
        situation.append("The ball is visible on the field.")
    else:
        situation.append("The ball is not visible in the current frame.")
    if positions.get("goalkeepers"):
        situation.append("Goalkeepers are guarding their positions.")
    if positions.get("referees"):
        situation.append("Referees are on the pitch.")
    if positions.get("players"):
        situation.append("Players are actively moving across the field.")

    prompt = (
        "You are a football commentator. "
        "Give a short, intelligent commentary based ONLY on the current frame. "
        "Do not repeat words or refer to other matches. "
        "Here is the frame description: "
        + " ".join(situation)
        + " Commentary:"
    )

    # Generate
    generated = commentator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.3
    )[0]['generated_text']

    # Clean result
    commentary = generated.replace(prompt, "").strip()

    # Remove looping phrases
    while any(phrase for phrase in ["goal-kicking", "goal is", "ball is"] if commentary.count(phrase) > 3):
        commentary = commentary.rsplit(".", 1)[0].strip() + "."

    # Fallback if still repeated
    if commentary == last_commentary or not commentary or len(set(commentary.split())) < 5:
        commentary = "A tense moment on the pitch as both teams hold their ground."

    last_commentary = commentary
    return commentary
