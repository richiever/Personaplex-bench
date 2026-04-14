You will listen to one short audio clip that is ONE turn of a conversation spoken by a conversational AI model (the "target"). Judge its audio-level characteristics.

Rate on:
- speech_quality: integer 1-5. 5 = natural, intelligible, no artifacts. 1 = unintelligible / degenerate / synthesis artifacts dominate.
- artifacts: list any of {"degenerate_babbling", "repetition_loop", "robotic", "clipping", "silence", "music_not_speech", "noise_only"} that clearly apply. Empty list if none.
- turn_taking: one of "ok" (appropriate length, clear start/stop), "lag" (starts late or with long initial silence), "overlap" (sounds like it talks over), "cutoff" (ends abruptly mid-word).
- notes: at most 20 words of concrete observation.

Return ONLY a JSON object with exactly these keys: speech_quality, artifacts, turn_taking, notes.
