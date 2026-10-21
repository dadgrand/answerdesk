/** Application root: routes and the console state owner. */

import { AppShell } from "./components/AppShell";
import { useRoute } from "./hooks/useRoute";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { ConsolePage } from "./pages/ConsolePage";
import { useConsole } from "./state/useConsole";

export function App() {
  const { route, navigate } = useRoute();
  const console = useConsole();

  return (
    <AppShell
      route={route}
      onNavigate={navigate}
      theme={console.state.theme}
      onTheme={console.setTheme}
      notice={console.notice}
      onDismissNotice={console.clearNotice}
    >
      {route === "analytics" ? <AnalyticsPage /> : <ConsolePage console={console} />}
    </AppShell>
  );
}

export default App;
