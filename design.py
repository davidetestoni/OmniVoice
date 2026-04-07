from omnivoice import OmniVoice
import torch
import torchaudio

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
)

# Valid English items: american accent, australian accent, british accent, canadian accent,
# child, chinese accent, elderly, female, high pitch, indian accent, japanese accent,
# korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent,
# russian accent, teenager, very high pitch, very low pitch, whisper, young adult
for i in range(5):
    audio = model.generate(
        text="Ciao, come stai oggi?",
        instruct="female, young adult, high pitch",
        language="it",
    )

    torchaudio.save(f"out_{i}.wav", audio[0], 24000)
