export function joinTranscriptTimelines(prefix: string, current: string): string {
  return [prefix.trim(), current.trim()].filter(Boolean).join("\n");
}
