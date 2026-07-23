import assert from "node:assert/strict";
import test from "node:test";

import { joinTranscriptTimelines } from "../src/transcript.ts";

test("preserves the transcript prefix when backend STT restarts after reconnect", () => {
  const beforeReconnect = "interviewer: First question\ncandidate: First answer";
  const restartedTimeline = "interviewer: Follow-up question";

  assert.equal(
    joinTranscriptTimelines(beforeReconnect, restartedTimeline),
    `${beforeReconnect}\n${restartedTimeline}`,
  );
});

test("does not add blank lines for an empty prefix or current timeline", () => {
  assert.equal(joinTranscriptTimelines("", "interviewer: Hello"), "interviewer: Hello");
  assert.equal(joinTranscriptTimelines("candidate: Hi", ""), "candidate: Hi");
});
