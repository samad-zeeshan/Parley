# ADR 0003: Turn taking, streaming and barge-in

Status: accepted, 2026-09-25

## Context

A phone caller does not wait for the agent to finish. They answer early, correct themselves, or start
talking over a long list of options. The agent has to keep listening while it talks and stop when the
caller starts. It also has to know when the caller has finished, without a button.

The reference machine has no GPU. The speech model we chose (ADR 0001, faster-whisper small) is not a
streaming model and takes seconds per utterance on this CPU.

## Decision

Everything runs on 20 ms frames of 16 kHz mono audio.

1. Endpointing. An energy VAD with an adaptive noise floor (`speech/vad.py`) marks the start of speech
   after 3 loud frames (60 ms) and the end after 25 quiet frames (500 ms hangover). 300 ms of audio
   before the start is kept so the first syllable is not cut.
2. Streaming ASR. `speech/asr.py` buffers the utterance while the VAD says speech, optionally re-decodes
   the growing buffer every N seconds for partial transcripts, and decodes the whole utterance once when
   the VAD closes it. This is "streaming" in the sense that the caller never presses a button and the
   decode starts the moment they stop. It is not incremental decoding.
3. Barge-in. `speech/duplex.py` runs the VAD on every mic frame while the agent plays. If the VAD
   reports the start of speech during playback, playback stops before the next frame. So the agent stops
   80 ms after the caller's first voiced frame on an exact clock: 60 ms to confirm onset, 20 ms for the
   frame in flight.
4. Echo. With no echo canceller, the agent's own voice reaches the mic. While playing, a mic frame counts
   as caller speech only if it is louder than the expected echo (the outgoing frame times an assumed
   speaker-to-mic gain of 0.3) by 6 dB. In the browser demo the browser's own echo cancellation is on.

## The bound we test

`BARGE_IN_BOUND_S = 0.2`. `tests/test_duplex.py` checks it two ways:

- On a shared simulated clock, with the caller starting at several offsets, with and without the agent's
  echo in the mic: the agent stops within 0.1 s of the caller's first voiced frame, and never stops on
  its own echo alone.
- With a real-time player thread and a real-time mic thread, each paced at 20 ms: the agent stops within
  0.2 s of wall-clock time. On the reference machine it measured 0.055 s in five runs.

## What this does not cover

- No real microphone or speaker was used in any test. The mic stream is synthetic or TTS audio, and the
  echo path is a fixed gain with no delay or room reverb. A real echo path is louder at some frequencies
  and arrives late; the fixed-gain check would need calibration per device, or a proper echo canceller.
- Energy VAD cannot tell a caller from a door slam or a TV. A neural VAD (Silero) is the upgrade.
- After a barge-in, the agent stops talking and listens. It does not yet resume or summarize what it was
  saying. The dialogue state is unchanged, so the next turn simply continues from the caller's words.
- The 500 ms hangover is added to every turn's latency before the ASR even starts. It is the price of not
  cutting callers off between words; the evaluation reports it separately.
