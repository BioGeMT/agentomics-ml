import { highlight } from "@/model/highlight";
import { splitCommand } from "@/model/text-kinds";

const RUN_DIR_PREFIX = "/workspace/run/current_iteration/current_step/";

/** A shell command, one line per command in the chain, with the run folder prefix dimmed. */
export function ShellCommand({ command }: { command: string }) {
  const parts = splitCommand(command);
  return (
    <pre className="shell-command">
      {parts.map((part, i) => {
        const html = (highlight(part.text, "bash") ?? part.text).split(RUN_DIR_PREFIX).join(`<span class="dim-prefix">${RUN_DIR_PREFIX}</span>`);
        return (
          <span key={i} className="shell-line">
            <span className="shell-connector">{i === 0 ? "$" : part.connector}</span>
            <span className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
            {"\n"}
          </span>
        );
      })}
    </pre>
  );
}
