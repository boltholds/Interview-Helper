import assert from "node:assert/strict";
import test from "node:test";

import { DualAudioCapture, downsampleToPcm16, encodeAudioFrame } from "../src/audio.ts";

test("encodes interviewer and candidate source tags without changing PCM", () => {
  const pcm = new Uint8Array([0, 1, 254, 255]).buffer;
  assert.deepEqual([...new Uint8Array(encodeAudioFrame("interviewer", pcm))], [1, 0, 1, 254, 255]);
  assert.deepEqual([...new Uint8Array(encodeAudioFrame("candidate", pcm))], [2, 0, 1, 254, 255]);
});

test("downsamples browser float audio to 16 kHz PCM16", () => {
  const pcm = new Int16Array(
    downsampleToPcm16(new Float32Array([1, 1, -1, -1]), 32_000),
  );
  assert.deepEqual([...pcm], [32767, -32768]);
});

test("stops display video immediately and explains when display audio was not shared", async () => {
  let videoStopped = false;
  let microphoneRequested = false;
  const videoTrack = { stop: () => { videoStopped = true; } };
  const display = {
    getVideoTracks: () => [videoTrack],
    getAudioTracks: () => [],
    getTracks: () => [videoTrack],
  };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      mediaDevices: {
        getDisplayMedia: async () => display,
        getUserMedia: async () => {
          microphoneRequested = true;
          throw new Error("microphone should not be requested without display audio");
        },
      },
    },
  });

  const capture = new DualAudioCapture();
  await assert.rejects(
    capture.start(() => {}),
    /включите «Поделиться аудио»/,
  );
  assert.equal(videoStopped, true);
  assert.equal(microphoneRequested, false);
});

test("disconnects partial nodes and closes the context when setup fails before registration", async () => {
  let contextClosed = false;
  const disconnected = [];
  let displayAudioStopped = false;
  let microphoneStopped = false;
  const stream = (onStop) => ({
    getVideoTracks: () => [],
    getAudioTracks: () => [{ stop: onStop }],
    getTracks: () => [{ stop: onStop }],
  });
  const display = stream(() => { displayAudioStopped = true; });
  const microphone = stream(() => { microphoneStopped = true; });
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      mediaDevices: {
        getDisplayMedia: async () => display,
        getUserMedia: async () => microphone,
      },
    },
  });
  globalThis.AudioContext = class {
    state = "running";
    destination = {};
    audioWorklet = { addModule: async () => {} };
    createMediaStreamSource() {
      return {
        connect: () => {},
        disconnect: () => { disconnected.push("source"); },
      };
    }
    createGain() {
      return {
        gain: { value: 1 },
        connect: () => { throw new Error("destination failed"); },
        disconnect: () => { disconnected.push("gain"); },
      };
    }
    async close() {
      contextClosed = true;
      this.state = "closed";
    }
  };
  globalThis.AudioWorkletNode = class {
    port = {};
    connect() {}
    disconnect() { disconnected.push("worklet"); }
  };

  const capture = new DualAudioCapture();
  await assert.rejects(capture.start(() => {}), /destination failed/);
  assert.deepEqual(disconnected, ["gain", "worklet", "source"]);
  assert.equal(contextClosed, true);
  assert.equal(displayAudioStopped, true);
  assert.equal(microphoneStopped, true);
});
