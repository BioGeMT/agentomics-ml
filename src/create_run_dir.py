import os
import sys
import random
import subprocess

def create_run_user(run_id):
    run_dir = os.path.join("/workspace/runs", run_id)

    
    # Create the user with home directory set to run_dir
    subprocess.run(
        ["sudo", "useradd", "-d", run_dir, "-m", "-p", "1234", run_id], #TODO use --password so other agents can't switch to it
        check=True
    )
    
    # Set ownership of the run directory to the new user
    # subprocess.run(
    #     ["sudo", "chown", "-R", f"{run_id}:{run_id}", run_dir],
    #     check=True
    # )
    
    print(f"Created user {run_id} for run {run_id}")
    return run_id

def generate_readable_id():
    """
    Generate a readable ID like 'serene-pyramid-26' or 'unique-yogurt-28'.
    
    Returns:
        A readable ID string
    """
    adjectives = [
        "autumn", "hidden", "bitter", "misty", "silent", "empty", "dry", "dark",
        "summer", "icy", "delicate", "quiet", "white", "cool", "spring", "winter",
        "patient", "twilight", "dawn", "crimson", "wispy", "weathered", "blue",
        "billowing", "broken", "cold", "damp", "falling", "frosty", "green",
        "long", "late", "lingering", "bold", "little", "morning", "muddy", "old",
        "red", "rough", "still", "small", "sparkling", "throbbing", "shy",
        "wandering", "withered", "wild", "black", "young", "holy", "solitary",
        "fragrant", "aged", "snowy", "proud", "floral", "restless", "divine",
        "polished", "ancient", "purple", "lively", "nameless", "lucky", "odd", "tiny",
        "free", "dry", "yellow", "orange", "gentle", "tight", "super", "royal", "broad",
        "steep", "flat", "square", "round", "mute", "noisy", "hushy", "raspy", "soft",
        "shrill", "rapid", "sweet", "curly", "calm", "jolly", "fancy", "plain", "shinny"
    ]

    nouns = [
        "waterfall", "river", "breeze", "moon", "rain", "wind", "sea", "morning",
        "snow", "lake", "sunset", "pine", "shadow", "leaf", "dawn", "glitter",
        "forest", "hill", "cloud", "meadow", "sun", "glade", "bird", "brook",
        "butterfly", "bush", "dew", "dust", "field", "fire", "flower", "firefly",
        "feather", "grass", "haze", "mountain", "night", "pond", "darkness",
        "snowflake", "silence", "sound", "sky", "shape", "surf", "thunder",
        "violet", "water", "wildflower", "wave", "water", "resonance", "sun",
        "wood", "dream", "cherry", "tree", "fog", "frost", "voice", "paper",
        "frog", "smoke", "star", "atom", "band", "bar", "base", "block", "boat",
        "term", "credit", "art", "feeling", "hero", "hope", "idea", "king",
        "line", "lab", "material", "math", "moment", "nation", "page", "park", "party",
        "pattern", "piano", "plant", "scene", "soil", "square", "stadium", "thing",
        "texture", "kitchen", "island", "earth", "field", "hotel", "ice", "jungle",
        "light", "market", "ocean", "plane", "road", "spring", "star", "summer",
        "autumn", "train", "water", "winter", "desert", "dawn", "dusk", "forest"
    ]

    # Choose a random adjective and noun
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    
    # Add a random number between 0 and 99
    number = random.randint(0, 999)
    
    # Combine to form the ID
    return f"{adjective}-{noun}-{number}"

def create_run_directory():
    run_id = generate_readable_id()
        
    username = create_run_user(run_id)
    
    print(f"Run ID: {run_id}")
    print(f"Created user {username}")
    return run_id

create_run_directory()