/** The conversation pane: transcript, customer context and the suggested answer. */

import type { Conversation } from "../demo/inbox";
import { formatClock, formatUsd, shortId } from "../lib/format";
import { slaFor } from "../lib/sla";
import type { Rating, SentReply, TurnState } from "../state/console";
import { AnswerCard } from "./AnswerCard";
import { LanguageBadge, PriorityDot } from "./Badges";

export interface ConversationViewProps {
  conversation: Conversation;
  turn: TurnState | null;
  now: number;
  sent: SentReply | null;
  feedback: Rating | null;
  ticket: { ticketId: string; reason: string } | null;
  onSend: (text: string) => void;
  onEscalate: () => void;
  onRate: (rating: Rating) => void;
  onOpenDiagnostics: () => void;
  onRetry: () => void;
}

export function ConversationView(props: ConversationViewProps) {
  const { conversation, turn, now, sent } = props;
  const sla = slaFor(conversation.waitingSince, now);

  return (
    <section className="conversation" aria-labelledby="conversation-title">
      <header className="conversation__head">
        <div className="conversation__person">
          <span className="avatar avatar--lg" aria-hidden="true">
            {conversation.customer.initials}
          </span>
          <div>
            <h2 id="conversation-title" className="conversation__name">
              {conversation.customer.name}
            </h2>
            <p className="conversation__meta">
              <LanguageBadge locale={conversation.customer.locale} />
              <span>{conversation.customer.plan}</span>
              <span>·</span>
              <span>{conversation.customer.tenure} with us</span>
              <span>·</span>
              <span>{conversation.channel}</span>
            </p>
          </div>
        </div>
        <dl className="conversation__facts">
          <div>
            <dt>Intent</dt>
            <dd>{conversation.intentLabel ?? conversation.intentBucket}</dd>
          </div>
          <div>
            <dt>Risk tier</dt>
            <dd>{conversation.intentBucket.replace("_", " / ")}</dd>
          </div>
          <div>
            <dt>Waiting</dt>
            <dd className={`conversation__sla conversation__sla--${sla.state}`}>
              {formatClock(sla.ageSeconds)} {sla.state === "breach" ? "· breached" : sla.state === "warn" ? "· at risk" : ""}
            </dd>
          </div>
          <div>
            <dt>Session</dt>
            <dd title={`s-console-${conversation.id}`}>{shortId(`s-console-${conversation.id}`, 10, 4)}</dd>
          </div>
        </dl>
      </header>

      <ol className="thread" aria-label="Conversation transcript">
        {conversation.messages.map((message, index) => (
          <li key={`${conversation.id}-m${index}`} className={`bubble bubble--${message.role}`}>
            <div className="bubble__meta">
              <span>{message.role === "customer" ? conversation.customer.name.split(" ")[0] : "Operator"}</span>
              <span aria-hidden="true">·</span>
              <span>{formatClock((now - message.at) / 1000)} ago</span>
              {index === 0 ? <PriorityDot priority={conversation.priority} /> : null}
            </div>
            <p className="bubble__text" dir="auto" lang={conversation.customer.locale}>
              {message.text}
            </p>
          </li>
        ))}
        {sent ? (
          <li className="bubble bubble--agent">
            <div className="bubble__meta">
              <span>Operator · you</span>
              <span aria-hidden="true">·</span>
              <span>just now</span>
            </div>
            <p className="bubble__text" dir="auto">
              {sent.text}
            </p>
          </li>
        ) : null}
      </ol>

      <AnswerCard
        conversation={conversation}
        turn={turn}
        sentText={sent?.text ?? null}
        feedback={props.feedback}
        ticket={props.ticket}
        onSend={props.onSend}
        onEscalate={props.onEscalate}
        onRate={props.onRate}
        onOpenDiagnostics={props.onOpenDiagnostics}
        onRetry={props.onRetry}
      />

      {turn?.usage ? (
        <footer className="conversation__usage" aria-label="Turn cost and latency">
          <span>{turn.usage.model}</span>
          <span>{turn.usage.input_tokens + turn.usage.output_tokens} tokens</span>
          <span>{formatUsd(turn.usage.usd, 5)}</span>
          <span>
            {turn.answer?.cache.status === "hit"
              ? "served from the answer cache"
              : turn.timings.total
                ? `${Math.round(turn.timings.total)} ms server-side`
                : ""}
          </span>
          <span className="conversation__trace" title={turn.traceId ?? ""}>
            trace {turn.traceId ? shortId(turn.traceId, 8, 6) : "—"}
          </span>
        </footer>
      ) : null}
    </section>
  );
}
