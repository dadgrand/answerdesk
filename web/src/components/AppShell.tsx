/** The application shell: navigation rail, top bar with service identity, content, toast. */

import { useEffect, type ReactNode } from "react";

import { mockRequested } from "../mock";
import { useServiceInfo } from "../hooks/useServiceInfo";
import type { Route } from "../hooks/useRoute";
import type { Notice } from "../state/useConsole";
import type { Theme } from "../state/console";
import { shortId } from "../lib/format";

function ServiceBadge() {
  const service = useServiceInfo();
  if (service.status === "loading") {
    return (
      <span className="status-chip status-chip--loading" title="Probing /v1/info">
        <span className="status-dot" /> connecting…
      </span>
    );
  }
  if (service.status === "error") {
    return (
      <span className="status-chip status-chip--bad" title={service.message}>
        <span className="status-dot" /> service unreachable
      </span>
    );
  }
  const { info, health } = service;
  const degraded = health?.status === "degraded";
  return (
    <span
      className={`status-chip ${degraded ? "status-chip--warn" : "status-chip--ok"}`}
      title={`${info.service} ${info.version} · ${info.profile} · ${info.corpus_version} · ${info.index_version}`}
    >
      <span className="status-dot" />
      <span className="status-chip__model">{info.model}</span>
      <span className="status-chip__sep" aria-hidden="true">
        ·
      </span>
      <span className="status-chip__meta">
        {info.chunks} chunks · {shortId(info.corpus_version, 12, 4)}
      </span>
      {degraded ? <span className="status-chip__flag">degraded</span> : null}
    </span>
  );
}

export interface AppShellProps {
  route: Route;
  onNavigate: (route: Route) => void;
  theme: Theme;
  onTheme: (theme: Theme) => void;
  notice: Notice | null;
  onDismissNotice: () => void;
  children: ReactNode;
}

const TITLES: Record<Route, { title: string; subtitle: string }> = {
  console: {
    title: "Operator console",
    subtitle: "Grounded answers · verified citations · calibrated escalation",
  },
  analytics: {
    title: "Analytics",
    subtitle: "Latency, cost, escalation and cache series from the service",
  },
};

export function AppShell(props: AppShellProps) {
  const { route, onNavigate, theme, onTheme, notice, onDismissNotice } = props;
  const copy = TITLES[route];

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(onDismissNotice, 6000);
    return () => window.clearTimeout(timer);
  }, [notice, onDismissNotice]);

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <nav className="rail" aria-label="Primary">
        <div className="rail__brand" aria-hidden="true">
          a<span className="rail__brand-dot">.</span>
        </div>
        <div className="rail__nav">
          <button
            type="button"
            className={`rail__link ${route === "console" ? "rail__link--active" : ""}`}
            aria-current={route === "console" ? "page" : undefined}
            onClick={() => onNavigate("console")}
            title="Console"
          >
            <span className="rail__icon" aria-hidden="true">
              ◫
            </span>
            <span className="rail__label">Console</span>
          </button>
          <button
            type="button"
            className={`rail__link ${route === "analytics" ? "rail__link--active" : ""}`}
            aria-current={route === "analytics" ? "page" : undefined}
            onClick={() => onNavigate("analytics")}
            title="Analytics"
          >
            <span className="rail__icon" aria-hidden="true">
              ◴
            </span>
            <span className="rail__label">Analytics</span>
          </button>
        </div>
        <div className="rail__spacer" />
        <button
          type="button"
          className="rail__link rail__link--icon"
          onClick={() => onTheme(theme === "dark" ? "light" : "dark")}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          <span className="rail__icon" aria-hidden="true">
            {theme === "dark" ? "☾" : "☀"}
          </span>
        </button>
      </nav>

      <div className="workspace">
        <header className="topbar">
          <div className="topbar__titles">
            <h1 className="topbar__title">{copy.title}</h1>
            <p className="topbar__sub">{copy.subtitle}</p>
          </div>
          <div className="topbar__tools">
            {mockRequested() ? (
              <span className="chip chip--mock" title="MSW mock mode: requests are served by src/api/mock/handlers.ts">
                mock mode
              </span>
            ) : null}
            <ServiceBadge />
            <button
              type="button"
              className="button button--ghost topbar__theme"
              onClick={() => onTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? "☀ Light" : "☾ Dark"}
            </button>
          </div>
        </header>

        <main className="main" id="main">
          {props.children}
        </main>
      </div>

      {notice ? (
        <div className={`toast toast--${notice.kind}`} role="status" aria-live="polite">
          <span>{notice.message}</span>
          <button type="button" className="icon-button" onClick={onDismissNotice} aria-label="Dismiss notification">
            ×
          </button>
        </div>
      ) : null}
    </div>
  );
}
