/** Shown when the lobby is opened in a browser instead of the desktop app. */
export function OpenDesktopApp() {
  return (
    <div className="lobby lobby-browser">
      <main className="lobby-main">
        <div className="lobby-content">
          <h1>Open the desktop app</h1>
          <p className="lobby-lead">
            Starting runs needs the Agentomics desktop app, which picks dataset folders and runs Docker. Start it with{" "}
            <code>cd ui/desktop &amp;&amp; npm start</code>.
          </p>
        </div>
      </main>
    </div>
  );
}
