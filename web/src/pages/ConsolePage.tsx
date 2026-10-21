/** Console screen: queue + conversation + diagnostics drawer. */

import { DiagnosticsDrawer } from "../components/DiagnosticsDrawer";
import { ConversationView } from "../components/ConversationView";
import { EmptyState } from "../components/Badges";
import { QueuePanel } from "../components/QueuePanel";
import type { UseConsole } from "../state/useConsole";

export function ConsolePage({ console }: { console: UseConsole }) {
  const { state, conversation, turn, now } = console;

  return (
    <div className={`console ${state.diagnosticsOpen ? "console--drawer" : ""}`}>
      <QueuePanel
        conversations={console.conversations}
        selectedId={state.selectedId}
        now={now}
        turns={state.turns}
        localeFilter={state.localeFilter}
        slaFilter={state.slaFilter}
        onSelect={console.selectConversation}
        onLocaleFilter={console.setLocaleFilter}
        onSlaFilter={console.setSlaFilter}
      />

      {conversation ? (
        <ConversationView
          conversation={conversation}
          turn={turn}
          now={now}
          sent={state.sent[conversation.id] ?? null}
          feedback={state.feedback[conversation.id] ?? null}
          ticket={state.tickets[conversation.id] ?? null}
          onSend={(text) => console.sendDraft(conversation.id, text)}
          onEscalate={() => void console.escalate(conversation.id)}
          onRate={(rating) => void console.rate(conversation.id, rating)}
          onOpenDiagnostics={() => console.toggleDiagnostics(true)}
          onRetry={() => console.retryTurn(conversation.id)}
        />
      ) : (
        <div className="conversation conversation--empty">
          <EmptyState title="Select a conversation" hint="The queue on the left lists every waiting customer." />
        </div>
      )}

      <DiagnosticsDrawer
        open={state.diagnosticsOpen}
        traceId={turn?.traceId ?? null}
        retrieval={turn?.retrieval ?? null}
        onClose={() => console.toggleDiagnostics(false)}
        loadTrace={console.loadTrace}
      />

      {state.diagnosticsOpen ? (
        <button
          type="button"
          className="drawer-backdrop"
          aria-label="Close diagnostics"
          onClick={() => console.toggleDiagnostics(false)}
        />
      ) : null}
    </div>
  );
}
