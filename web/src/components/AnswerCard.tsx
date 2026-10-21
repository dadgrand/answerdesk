/**
 * The suggested answer: verified text with inline citation chips, confidence, decision
 * state and the operator actions.
 *
 * Two product rules are visible here:
 *
 * * **No citation chip without a verified citation.** While the answer streams, `[1]`
 *   markers render as plain text; the chips become interactive only after the `citations`
 *   event, which the service emits after verification (INV-02/INV-03). The UI therefore
 *   cannot display a citation the verifier did not produce.
 * * **An escalation is a first-class result.** When the decision is not `answer`, the card
 *   shows the reason code, the detail and the ticket instead of an empty answer box.
 */

import { useEffect, useId, useMemo, useRef, useState } from "react";

import type { Citation, Decision } from "../api/types";
import type { Conversation } from "../demo/inbox";
import { formatPct, humanizeReason, shortId } from "../lib/format";
import type { Rating, TurnState } from "../state/console";
import { CacheChip, ConfidenceMeter, DecisionBadge, ErrorBanner } from "./Badges";

export interface AnswerCardProps {
  conversation: Conversation;
  turn: TurnState | null;
  sentText: string | null;
  feedback: Rating | null;
  ticket: { ticketId: string; reason: string } | null;
  onSend: (text: string) => void;
  onEscalate: () => void;
  onRate: (rating: Rating) => void;
  onOpenDiagnostics: () => void;
  onRetry: () => void;
}

function stageLabel(turn: TurnState | null): string {
  if (!turn) return "Waiting for the service…";
  if (turn.status === "streaming") {
    if (!turn.retrieval) return "Retrieving evidence…";
    if (turn.citations.length === 0) return "Drafting and verifying claims…";
    return "Verifying citations…";
  }
  if (turn.status === "complete") return "Verified answer";
  if (turn.status === "escalated") return "Escalated";
  if (turn.status === "error") return "Failed";
  return "Idle";
}

function decisionOf(turn: TurnState | null): Decision | null {
  if (!turn) return null;
  if (turn.answer) return turn.answer.decision;
  if (turn.decision) {
    return {
      action: turn.decision.action,
      reason_code: turn.decision.reason_code,
      score: turn.decision.score,
      alpha: turn.decision.alpha,
      threshold_ref: turn.decision.threshold_ref,
    };
  }
  return null;
}

/**
 * A plain-language explanation of *why* the policy did not answer.
 *
 * The service exposes the reason code, p̂ and α; the console turns them into the sentence an
 * operator can act on. It never invents a reason the decision does not carry.
 */
export function policyExplanation(decision: Decision | null, intentBucket: string): string {
  if (!decision) return "The policy engine did not return a decision for this turn.";
  const required = 1 - decision.alpha;
  switch (decision.reason_code) {
    case "PII_DETECTED":
      return "The PII guard fired before the model call: customer data is never sent for an auto-answer.";
    case "INJECTION_SUSPECTED":
      return "Retrieved content tried to change the task; the turn is handed to a human (INV-01).";
    case "NO_EVIDENCE":
      return "No passage passed the evidence threshold — the knowledge base cannot support an answer.";
    case "CITATION_FAILED":
      return "A citation or the numeric guard failed verification; the draft was withheld (INV-02/INV-04).";
    case "VERIFIER_REJECT":
      return "Every claim failed verification, so nothing could be displayed (INV-03).";
    case "HIGH_RISK_INTENT":
      return "This intent is on the high-risk list (account, legal, payment) and always goes to a human.";
    case "STALE_CALIBRATION":
      return "The calibration artifact is stale, so the service refuses to answer as if it were calibrated.";
    case "LOW_CONFIDENCE":
      return `p̂ ${formatPct(decision.score)} is below the ${formatPct(required)} band required for ${
        intentBucket.replace("_", " / ")
      } turns.`;
    default:
      return humanizeReason(decision.reason_code);
  }
}

export function CitationChip({
  citation,
  active,
  onSelect,
}: {
  citation: Citation;
  active: boolean;
  onSelect: () => void;
}) {
  const tooltipId = useId();
  return (
    <span className="cite-wrap">
      <button
        type="button"
        className={`cite ${active ? "cite--active" : ""}`}
        aria-label={`Citation ${citation.citation_id}: ${citation.doc_title}. Show the source passage.`}
        aria-expanded={active}
        aria-describedby={tooltipId}
        onClick={onSelect}
      >
        {citation.citation_id}
      </button>
      <span className="cite__tip" id={tooltipId} role="tooltip">
        <span className="cite__tip-title">{citation.doc_title}</span>
        <span className="cite__tip-heading">{citation.heading_path.join(" › ")}</span>
        <span className="cite__tip-quote">“{citation.quote}”</span>
      </span>
    </span>
  );
}

export function PassageCard({
  citation,
  onClose,
  onOpenDiagnostics,
}: {
  citation: Citation;
  onClose: () => void;
  onOpenDiagnostics: () => void;
}) {
  return (
    <aside className="passage" aria-label={`Source passage for citation ${citation.citation_id}`}>
      <header className="passage__head">
        <span className="passage__kicker">Source passage · citation {citation.citation_id}</span>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close the source passage">
          ×
        </button>
      </header>
      <h4 className="passage__title">{citation.doc_title}</h4>
      <p className="passage__heading">{citation.heading_path.join(" › ")}</p>
      <blockquote className="passage__quote">“{citation.quote}”</blockquote>
      <dl className="passage__meta">
        <div>
          <dt>Chunk</dt>
          <dd title={citation.chunk_id}>{citation.chunk_id}</dd>
        </div>
        <div>
          <dt>Span</dt>
          <dd>
            {citation.start_char}–{citation.end_char}
          </dd>
        </div>
        <div>
          <dt>Doc hash</dt>
          <dd title={citation.doc_hash}>{shortId(citation.doc_hash, 8, 4)}</dd>
        </div>
      </dl>
      <footer className="passage__foot">
        <button type="button" className="button button--ghost" onClick={onOpenDiagnostics}>
          Open retrieval diagnostics
        </button>
      </footer>
    </aside>
  );
}

function renderWithCitations(
  text: string,
  citations: Citation[],
  activeId: number | null,
  onSelect: (id: number) => void,
): React.ReactNode[] {
  const known = new Map(citations.map((citation) => [citation.citation_id, citation]));
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) return <span key={`t-${index}`}>{part}</span>;
    const id = Number(match[1]);
    const citation = known.get(id);
    if (!citation) {
      // Unverified marker: render as text until the verifier produces the citation.
      return (
        <span key={`u-${index}`} className="cite-pending">
          {part}
        </span>
      );
    }
    return (
      <CitationChip
        key={`c-${id}-${index}`}
        citation={citation}
        active={activeId === id}
        onSelect={() => onSelect(id)}
      />
    );
  });
}

export function AnswerCard(props: AnswerCardProps) {
  const { conversation, turn, sentText, feedback, ticket } = props;
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [edited, setEdited] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const decision = decisionOf(turn);
  const answerText = edited ?? turn?.answer?.text ?? turn?.streamingText ?? "";
  const citations = turn?.citations.length ? turn.citations : (turn?.answer?.citations ?? []);
  const status = turn?.status ?? "idle";
  const streaming = status === "streaming";
  const isAnswer = decision?.action === "answer" && status === "complete";
  const isEscalated = status === "escalated" || (decision !== null && decision.action !== "answer" && status !== "error");
  const openTicket = ticket?.ticketId ?? turn?.ticketId ?? null;
  const selectedCitation = useMemo(
    () => citations.find((citation) => citation.citation_id === activeCitation) ?? null,
    [citations, activeCitation],
  );

  useEffect(() => {
    if (editing) return;
    setDraft(answerText);
  }, [answerText, editing]);

  useEffect(() => {
    if (activeCitation !== null && !citations.some((citation) => citation.citation_id === activeCitation)) {
      setActiveCitation(null);
    }
  }, [citations, activeCitation]);

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape" && activeCitation !== null) {
      setActiveCitation(null);
    }
  };

  const send = () => {
    props.onSend(edited ?? answerText);
  };

  return (
    <section
      className={`answer answer--${status}`}
      aria-labelledby="answer-title"
      onKeyDown={onKeyDown}
      ref={containerRef}
    >
      <header className="answer__head">
        <div className="answer__title-wrap">
          <h3 id="answer-title" className="answer__title">
            Suggested reply
          </h3>
          <span className="answer__status" role="status" aria-live="polite">
            {streaming ? <span className="answer__pulse" aria-hidden="true" /> : null}
            {stageLabel(turn)}
            {streaming && turn ? <span className="answer__tokens">{turn.tokenCount} tokens</span> : null}
          </span>
        </div>
        <div className="answer__chips">
          {decision ? <DecisionBadge decision={decision} /> : null}
          {turn ? <CacheChip status={turn.answer?.cache.status ?? turn.retrieval?.cache.status ?? "miss"} similarity={turn.answer?.cache.similarity} /> : null}
          {isAnswer ? <span className="chip chip--verified">✓ {citations.length} verified citations</span> : null}
        </div>
      </header>

      {turn?.error ? (
        <ErrorBanner title={`Request failed · ${turn.error.code}`} message={turn.error.message} onRetry={props.onRetry} />
      ) : null}

      <div className="answer__metrics">
        <ConfidenceMeter value={decision?.score ?? null} threshold={0.95} />
        {decision ? (
          <dl className="answer__policy">
            <div>
              <dt>α · risk tier</dt>
              <dd>{formatPct(decision.alpha, 0)}</dd>
            </div>
            <div>
              <dt>threshold</dt>
              <dd title={decision.threshold_ref}>{decision.threshold_ref}</dd>
            </div>
            <div>
              <dt>trace</dt>
              <dd title={turn?.traceId ?? ""}>{turn?.traceId ? shortId(turn.traceId, 8, 6) : "—"}</dd>
            </div>
          </dl>
        ) : null}
      </div>

      {editing ? (
        <div className="answer__editor">
          <label className="visually-hidden" htmlFor="draft-editor">
            Edit the reply
          </label>
          <textarea
            id="draft-editor"
            className="answer__textarea"
            value={draft}
            rows={5}
            onChange={(event) => setDraft(event.target.value)}
          />
          <div className="answer__editor-actions">
            <button
              type="button"
              className="button button--primary"
              onClick={() => {
                setEdited(draft);
                setEditing(false);
              }}
            >
              Save edit
            </button>
            <button type="button" className="button button--ghost" onClick={() => setEditing(false)}>
              Cancel
            </button>
            <span className="answer__hint">Citation markers like [1] stay linked to their passage.</span>
          </div>
        </div>
      ) : (
        <div className={`answer__body ${streaming ? "answer__body--streaming" : ""}`} aria-busy={streaming}>
          {answerText ? (
            <p className="answer__text">
              {renderWithCitations(answerText, citations, activeCitation, setActiveCitation)}
              {streaming ? <span className="caret" aria-hidden="true" /> : null}
            </p>
          ) : turn && !isEscalated && !turn.error ? (
            <p className="answer__skeleton">
              <span className="skeleton-line" />
              <span className="skeleton-line skeleton-line--short" />
            </p>
          ) : null}
          {edited !== null && !sentText ? <span className="tag tag--edited">edited by operator</span> : null}
          {sentText ? <span className="tag tag--sent">✓ sent · marked resolved</span> : null}
        </div>
      )}

      {selectedCitation ? (
        <PassageCard
          citation={selectedCitation}
          onClose={() => setActiveCitation(null)}
          onOpenDiagnostics={props.onOpenDiagnostics}
        />
      ) : null}

      {isAnswer && turn?.claims.length ? (
        <section className="claims" aria-label="Verified claims">
          <h4 className="claims__title">
            Claims verified · {turn.claims.length}
            <span className="claims__hint">every displayed claim carries a verified citation (INV-03)</span>
          </h4>
          <ol className="claims__list">
            {turn.claims.map((claim) => (
              <li key={claim.claim_id} className="claims__item">
                <span className={`claims__support claims__support--${claim.support}`}>✓ {claim.support}</span>
                <span className="claims__text">{claim.text}</span>
                <span className="claims__cites">
                  {claim.citation_ids.map((citationId) => (
                    <button
                      key={`${claim.claim_id}-${citationId}`}
                      type="button"
                      className="cite cite--small"
                      aria-label={`Show citation ${citationId} for claim ${claim.claim_id}`}
                      onClick={() => setActiveCitation(citationId)}
                    >
                      {citationId}
                    </button>
                  ))}
                </span>
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {isEscalated ? (
        <div className="escalation" role="group" aria-label="Escalation">
          <div className="escalation__head">
            <span className="escalation__icon" aria-hidden="true">
              ⚑
            </span>
            <div>
              <p className="escalation__title">
                {decision?.action === "refuse" ? "Refused automatically" : "Needs a human"} ·{" "}
                {humanizeReason(decision?.reason_code)}
              </p>
              <p className="escalation__detail">
                {turn?.error?.message ?? policyExplanation(decision, conversation.intentBucket)}
              </p>
            </div>
          </div>
          <dl className="escalation__meta">
            <div>
              <dt>p̂</dt>
              <dd>{decision ? formatPct(decision.score) : "—"}</dd>
            </div>
            <div>
              <dt>policy band</dt>
              <dd>{decision ? `${formatPct(1 - decision.alpha)} required` : "—"}</dd>
            </div>
            <div>
              <dt>ticket</dt>
              <dd>{openTicket ?? "not created"}</dd>
            </div>
          </dl>
          <p className="escalation__note">
            The service never auto-answers this turn. Reply manually, or hand it to the next tier with one click.
          </p>
        </div>
      ) : null}

      <footer className="answer__actions">
        {!isEscalated ? (
          <button
            type="button"
            className="button button--primary"
            onClick={send}
            disabled={!answerText || status !== "complete" || Boolean(sentText)}
          >
            Send reply
          </button>
        ) : (
          <button
            type="button"
            className="button button--primary"
            onClick={() => {
              setEditing(true);
              setDraft("");
            }}
          >
            Write a reply
          </button>
        )}
        <button
          type="button"
          className="button button--ghost"
          onClick={() => {
            setEditing((value) => !value);
            setDraft(answerText);
          }}
          disabled={streaming || Boolean(sentText)}
        >
          {editing ? "Close editor" : "Edit"}
        </button>
        <button
          type="button"
          className="button button--danger"
          onClick={props.onEscalate}
          disabled={!turn?.traceId || Boolean(openTicket) || Boolean(sentText)}
          title={openTicket ? `Ticket ${openTicket} already open` : "Open an escalation ticket for this trace"}
        >
          {openTicket ? `Ticket ${shortId(openTicket, 7, 4)}` : "Escalate"}
        </button>
        <span className="answer__spacer" />
        <div className="answer__feedback" role="group" aria-label="Rate this answer">
          <button
            type="button"
            className={`icon-button ${feedback === "up" ? "icon-button--active" : ""}`}
            aria-pressed={feedback === "up"}
            aria-label="Rate the answer as correct"
            onClick={() => props.onRate("up")}
            disabled={!turn?.traceId}
          >
            👍
          </button>
          <button
            type="button"
            className={`icon-button ${feedback === "down" ? "icon-button--active" : ""}`}
            aria-pressed={feedback === "down"}
            aria-label="Rate the answer as wrong"
            onClick={() => props.onRate("down")}
            disabled={!turn?.traceId}
          >
            👎
          </button>
        </div>
        <button type="button" className="button button--outline" onClick={props.onOpenDiagnostics} disabled={!turn?.traceId}>
          Diagnostics
        </button>
      </footer>

      <p className="answer__disclaimer">
        v1.0 has no outbound connector: “Send reply” records the operator action against the trace; the message is
        delivered by your helpdesk. Grounded answers only — every displayed claim carries a verified citation.
      </p>
      <span className="visually-hidden" data-testid="conversation-id">
        {conversation.id}
      </span>
    </section>
  );
}
