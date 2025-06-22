from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from extract_positions import find_ball_proximity

# Load TinyLlama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
commentator = pipeline("text-generation", model=model, tokenizer=tokenizer)

last_commentary = ""

def generate_commentary(positions):
    global last_commentary

    situation = []
    proximity_comment = None

    # Inject frame ID if provided
    frame_id = positions.get("frame_id", "unknown")

    # Handle ball and proximity
    if positions.get("ball"):
        situation.append("The ball is visible on the field.")
        entity, dist = find_ball_proximity(positions)
        if entity:
            entity_name = {
                "players": "a player",
                "goalkeepers": "a goalkeeper",
                "main_referees": "the main referee",
                "side_referees": "a side referee"
            }.get(entity, "someone")
            situation.append(f"The ball is close to {entity_name}.")

            # Predefined rule-based commentary
            if entity == "players":
                proximity_comment = "The game is tense — rapid passes are being exchanged."
            elif entity == "goalkeepers":
                proximity_comment = "Huge chance! The ball is near the goal."
            elif entity == "side_referees":
                proximity_comment = "Corner zone pressure — one team might take the lead!"
            elif entity == "main_referees":
                proximity_comment = "The ball is in midfield — it's a strategic setup."
    else:
        situation.append("The ball is not visible.")

    # Other field entities
    if positions.get("goalkeepers"):
        situation.append("Goalkeepers are in position.")
    if positions.get("main_referees"):
        situation.append("Main referee on pitch.")
    if positions.get("side_referees"):
        situation.append("Side referees watching closely.")
    if positions.get("staff_members"):
        situation.append("Staff members are near the sidelines.")
    if positions.get("players"):
        situation.append("Players are moving.")

    # Use handcrafted comment if available
    if proximity_comment:
        commentary = proximity_comment
    else:
        # Prompt for generation
        prompt = (
            f"You are a football commentator. Frame ID: {frame_id}. "
            "Describe ONLY what is happening in this specific frame. "
            "Do not reference past or future events. "
            "Avoid phrases like '10 seconds later' or 'fans are watching'. "
            "Here is the frame description: "
            + " ".join(situation)
            + " Commentary:"
        )

        # Generate commentary with controlled sampling
        generated = commentator(
            prompt,
            max_new_tokens=60,
            num_return_sequences=1,
            do_sample=True,
            top_k=30,
            top_p=0.85,
            temperature=0.7,
            repetition_penalty=2.0
        )[0]['generated_text']

        # Clean generated output
        commentary = generated.replace(prompt, "").strip()

        # Remove known repetitive or irrelevant phrases
        bad_phrases = [
            "10 seconds later", "in the next play", "live TV coverage",
            "watching from home", "match has started", "started and", "again and again"
        ]
        for phrase in bad_phrases:
            commentary = commentary.replace(phrase, "")

        # Remove redundant or repeated sentences
        lines = []
        for line in commentary.split('.'):
            line = line.strip()
            if line and line not in lines:
                lines.append(line)
        commentary = '. '.join(lines).strip()
        if commentary and not commentary.endswith('.'):
            commentary += '.'

        # Fallback if output is poor or identical to last
        if commentary == last_commentary or not commentary or len(set(commentary.split())) < 10:
            commentary = (
                "Players hold their ground as action continues. "
                "The teams are positioning themselves for the next move."
            )

    last_commentary = commentary
    return commentary
