export const TARGET_SAMPLE_RATE = 16_000;
const PROCESSOR_NAME = "interview-helper-pcm-capture";

export type AudioSource = "interviewer" | "candidate";
const SOURCE_TAG: Record<AudioSource, number> = { interviewer: 1, candidate: 2 };

const workletSource = `
class InterviewHelperPcmCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = new Float32Array(4096); this.offset = 0; }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    let sourceOffset = 0;
    while (sourceOffset < channel.length) {
      const count = Math.min(channel.length - sourceOffset, this.buffer.length - this.offset);
      this.buffer.set(channel.subarray(sourceOffset, sourceOffset + count), this.offset);
      this.offset += count; sourceOffset += count;
      if (this.offset === this.buffer.length) {
        const output = this.buffer.slice(); this.port.postMessage(output.buffer, [output.buffer]); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor("${PROCESSOR_NAME}", InterviewHelperPcmCapture);
`;

export function downsampleToPcm16(input: Float32Array, inputSampleRate: number): ArrayBuffer {
  if (inputSampleRate < TARGET_SAMPLE_RATE) {
    throw new Error(`Unsupported audio sample rate: ${inputSampleRate}`);
  }
  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const output = new Int16Array(Math.max(1, Math.floor(input.length / ratio)));
  for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
    const start = Math.floor(outputIndex * ratio);
    const end = Math.max(start + 1, Math.floor((outputIndex + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let inputIndex = start; inputIndex < end && inputIndex < input.length; inputIndex += 1) {
      sum += input[inputIndex];
      count += 1;
    }
    const sample = Math.max(-1, Math.min(1, count ? sum / count : 0));
    output[outputIndex] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output.buffer;
}

/** Dual-source v1 frames start with a one-byte source tag followed by PCM16 data. */
export function encodeAudioFrame(source: AudioSource, pcm16: ArrayBuffer): ArrayBuffer {
  const frame = new Uint8Array(pcm16.byteLength + 1);
  frame[0] = SOURCE_TAG[source];
  frame.set(new Uint8Array(pcm16), 1);
  return frame.buffer;
}

type AudioPipeline = {
  stream: MediaStream;
  context: AudioContext;
  source: MediaStreamAudioSourceNode;
  worklet: AudioWorkletNode;
  silentGain: GainNode;
};

export class DualAudioCapture {
  private pipelines: AudioPipeline[] = [];

  async start(onChunk: (source: AudioSource, pcm16: ArrayBuffer) => void): Promise<void> {
    if (this.pipelines.length) return;
    let display: MediaStream | null = null;
    let microphone: MediaStream | null = null;
    try {
      display = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      display.getVideoTracks().forEach((track) => track.stop());
      if (!display.getAudioTracks().length) {
        throw new Error(
          "В выбранной вкладке/окне нет аудио. Повторите выбор и включите «Поделиться аудио».",
        );
      }
      microphone = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      this.pipelines.push(await this.createPipeline(display, "interviewer", onChunk));
      this.pipelines.push(await this.createPipeline(microphone, "candidate", onChunk));
    } catch (error) {
      display?.getTracks().forEach((track) => track.stop());
      microphone?.getTracks().forEach((track) => track.stop());
      await this.stop();
      throw error;
    }
  }

  private async createPipeline(
    stream: MediaStream,
    sourceName: AudioSource,
    onChunk: (source: AudioSource, pcm16: ArrayBuffer) => void,
  ): Promise<AudioPipeline> {
    const context = new AudioContext();
    let source: MediaStreamAudioSourceNode | null = null;
    let worklet: AudioWorkletNode | null = null;
    let silentGain: GainNode | null = null;
    try {
      const moduleUrl = URL.createObjectURL(
        new Blob([workletSource], { type: "application/javascript" }),
      );
      try {
        await context.audioWorklet.addModule(moduleUrl);
      } finally {
        URL.revokeObjectURL(moduleUrl);
      }
      source = context.createMediaStreamSource(stream);
      worklet = new AudioWorkletNode(context, PROCESSOR_NAME, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      silentGain = context.createGain();
      silentGain.gain.value = 0;
      worklet.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        onChunk(sourceName, downsampleToPcm16(new Float32Array(event.data), context.sampleRate));
      };
      source.connect(worklet);
      worklet.connect(silentGain);
      silentGain.connect(context.destination);
      await context.resume();
      return { stream, context, source, worklet, silentGain };
    } catch (error) {
      for (const node of [silentGain, worklet, source]) {
        try {
          node?.disconnect();
        } catch {
          // Continue cleanup even if a partially connected browser node rejects it.
        }
      }
      if (context.state !== "closed") {
        await context.close();
      }
      throw error;
    }
  }

  async stop(): Promise<void> {
    const pipelines = this.pipelines;
    this.pipelines = [];
    await Promise.all(
      pipelines.map(async ({ stream, context, source, worklet, silentGain }) => {
        worklet.disconnect();
        source.disconnect();
        silentGain.disconnect();
        stream.getTracks().forEach((track) => track.stop());
        if (context.state !== "closed") await context.close();
      }),
    );
  }
}
