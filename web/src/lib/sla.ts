/** SLA arithmetic for the queue: age, target and state (pure, unit-tested). */

export const SLA_TARGET_SECONDS = 120;
export const SLA_WARN_SECONDS = 75;

export type SlaState = "ok" | "warn" | "breach";

export interface Sla {
  ageSeconds: number;
  state: SlaState;
  remainingSeconds: number;
  targetSeconds: number;
}

export function slaFor(waitingSinceMs: number, nowMs: number): Sla {
  const ageSeconds = Math.max(0, (nowMs - waitingSinceMs) / 1000);
  const remainingSeconds = Math.max(0, SLA_TARGET_SECONDS - ageSeconds);
  let state: SlaState = "ok";
  if (ageSeconds >= SLA_TARGET_SECONDS) state = "breach";
  else if (ageSeconds >= SLA_WARN_SECONDS) state = "warn";
  return { ageSeconds, state, remainingSeconds, targetSeconds: SLA_TARGET_SECONDS };
}

export function slaLabel(sla: Sla): string {
  if (sla.state === "breach") return "SLA breached";
  if (sla.state === "warn") return "SLA at risk";
  return "Within SLA";
}
