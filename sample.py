from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
)
# Apple Silicon users: use device_map="mps" instead

audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    #  ref_text="Transcription of the reference audio.",
)  # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

# If you don't want to input `ref_text` manually, you can directly omit the `ref_text`.
# The model will use Whisper ASR to auto-transcribe it.

sf.write("out.wav", audio[0], 24000)
