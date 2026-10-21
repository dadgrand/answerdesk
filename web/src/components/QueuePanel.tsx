/** The live queue: conversations with a language badge, priority and an SLA timer. */

import type { Conversation } from "../demo/inbox";
import { formatClock, languageName } from "../lib/format";
import { slaFor, slaLabel, type SlaState } from "../lib/sla";
import type { TurnState } from "../state/console";
import { EmptyState, LanguageBadge, PriorityDot } from "./Badges";

function SlaRing({ state, remaining, target }: { state: SlaState; remaining: number; target: number }) {
  const ratio = Math.max(0, Math.min(1, remaining / target));
  const radius = 8;
  const circumference = 2 * Math.PI * radius;
  return (
    <svg className={`sla-ring sla-ring--${state}`} width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <circle cx="11" cy="11" r={radius} className="sla-ring__track" fill="none" strokeWidth="2.5" />
      <circle
        cx="11"
        cy="11"
        r={radius}
        className="sla-ring__value"
        fill="none"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - ratio)}
        transform="rotate(-90 11 11)"
      />
    </svg>
  );
}

export interface QueuePanelProps {
  conversations: Conversation[];
  selectedId: string | null;
  now: number;
  turns: Record<string, TurnState>;
  localeFilter: "all" | string;
  slaFilter: "all" | "at-risk";
  onSelect: (conversationId: string) => void;
  onLocaleFilter: (locale: "all" | string) => void;
  onSlaFilter: (sla: "all" | "at-risk") => void;
}

const LOCALES = ["all", "ru", "tg", "uz", "ky", "tr", "zh"] as const;

export function QueuePanel(props: QueuePanelProps) {
  const { conversations, selectedId, now, turns } = props;

  return (
    <section className="queue" aria-labelledby="queue-title">
      <header className="queue__head">
        <div>
          <h2 id="queue-title" className="queue__title">
            Live queue
          </h2>
          <p className="queue__sub">{conversations.length} waiting · SLA target 2 min</p>
        </div>
        <div className="queue__filters" role="group" aria-label="Queue filters">
          <button
            type="button"
            className={`pill ${props.slaFilter === "all" ? "pill--active" : ""}`}
            aria-pressed={props.slaFilter === "all"}
            onClick={() => props.onSlaFilter("all")}
          >
            All
          </button>
          <button
            type="button"
            className={`pill ${props.slaFilter === "at-risk" ? "pill--active" : ""}`}
            aria-pressed={props.slaFilter === "at-risk"}
            onClick={() => props.onSlaFilter("at-risk")}
          >
            At risk
          </button>
        </div>
      </header>

      <label className="queue__locale">
        <span className="visually-hidden">Filter by language</span>
        <select
          value={props.localeFilter}
          onChange={(event) => props.onLocaleFilter(event.target.value)}
          aria-label="Filter by language"
        >
          {LOCALES.map((locale) => (
            <option key={locale} value={locale}>
              {locale === "all" ? "All languages" : languageName(locale)}
            </option>
          ))}
        </select>
      </label>

      {conversations.length === 0 ? (
        <EmptyState title="Nothing in this filter" hint="Clear the language or SLA filter to see the queue." />
      ) : (
        <ul className="queue__list" role="list">
          {conversations.map((conversation) => {
            const sla = slaFor(conversation.waitingSince, now);
            const turn = turns[conversation.id];
            const selected = conversation.id === selectedId;
            const latest = conversation.messages[conversation.messages.length - 1];
            return (
              <li key={conversation.id}>
                <button
                  type="button"
                  className={`queue__item ${selected ? "queue__item--selected" : ""}`}
                  aria-current={selected ? "true" : undefined}
                  onClick={() => props.onSelect(conversation.id)}
                >
                  <span className="queue__avatar" aria-hidden="true">
                    {conversation.customer.initials}
                  </span>
                  <span className="queue__body">
                    <span className="queue__row">
                      <span className="queue__name">{conversation.customer.name}</span>
                      <LanguageBadge locale={conversation.customer.locale} />
                      <PriorityDot priority={conversation.priority} />
                    </span>
                    <span className="queue__snippet">{latest?.text ?? conversation.subject}</span>
                    <span className="queue__meta">
                      <span className="queue__channel">{conversation.channel}</span>
                      {turn?.status === "escalated" ? <span className="tag tag--escalate">escalated</span> : null}
                      {turn?.status === "complete" ? <span className="tag tag--answer">answered</span> : null}
                      {turn?.status === "streaming" ? <span className="tag tag--stream">drafting</span> : null}
                      {turn?.status === "error" ? <span className="tag tag--error">failed</span> : null}
                    </span>
                  </span>
                  <span className={`queue__sla queue__sla--${sla.state}`} title={slaLabel(sla)}>
                    <SlaRing state={sla.state} remaining={sla.remainingSeconds} target={sla.targetSeconds} />
                    <span className="queue__clock">{formatClock(sla.ageSeconds)}</span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
