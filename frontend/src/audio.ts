const TARGET_SAMPLE_RATE = 16_000;
const PROCESSOR_NAME = "interview-helper-pcm-capture";

const workletSource = `
class InterviewHelperPcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(4096);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    let sourceOffset = 0;
    while (sourceOffset < channel.length) {
      const count = Math.min(channel.length - sourceOffset, this.buffer.length - this.offset);
      this.buffer.set(channel.subarray(sourceOffset, sourceOffset + count), this.offset);
      this.offset += count;
      sourceOffset += count;

      if (this.offset === this.buffer.length) {
        const output = this.buffer.slice();
        this.port.postMessage(output.buffer, [output.buffer]);
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor("${PROCESSOR_NAME}", InterviewHelperPcmCapture);
`;

function downsampleToPcm16(input: Float32Array, inputSampleRate: number): ArrayBuffer {
  if (inputSampleRate < TARGET_SAMPLE_RATE) {
    throw new Error(`Unsupported microphone sample rate: ${inputSampleRate}`);
  }

  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const outputLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Int16Array(outputLength);

  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const start = Math.floor(outputIndex * ratio);
    const end = Math.max(start + 1, Math.floor((outputIndex + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let inputIndex = start; inputIndex < end && inputIndex < input.length; inputIndex += 1) {
      sum += input[inputIndex];
      count += 1;
    }
    const sample = Math.max(-1, Math.min(1, count > 0 ? sum / count : 0));
    output[outputIndex] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }

  return output.buffer;
}

export class MicrophoneStreamer {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private worklet: AudioWorkletNode | null = null;
  private silentGain: GainNode | null = null;

  async start(onChunk: (pcm16: ArrayBuffer) => void): Promise<void> {
    if (this.context) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.context = new AudioContext();

    const blob = new Blob([workletSource], { type: "application/javascript" });
    const moduleUrl = URL.createObjectURL(blob);
    try {
      await this.context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }

    this.source = this.context.createMediaStreamSource(this.stream);
    this.worklet = new AudioWorkletNode(this.context, PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.silentGain = this.context.createGain();
    this.silentGain.gain.value = 0;

    const sourceSampleRate = this.context.sampleRate;
    this.worklet.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      const input = new Float32Array(event.data);
      onChunk(downsampleToPcm16(input, sourceSampleRate));
    };

    this.source.connect(this.worklet);
    this.worklet.connect(this.silentGain);
    this.silentGain.connect(this.context.destination);
    await this.context.resume();
  }

  async stop(): Promise<void> {
    this.worklet?.disconnect();
    this.source?.disconnect();
    this.silentGain?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.context && this.context.state !== "closed") {
      await this.context.close();
    }

    this.stream = null;
    this.context = null;
    this.source = null;
    this.worklet = null;
    this.silentGain = null;
  }
}
