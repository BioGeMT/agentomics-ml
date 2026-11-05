// Whether runs get the GPU. agentomics-run asks Docker for one unless told --cpu-only, and fails when Docker
// can't give it, so the app only asks where Docker has a GPU to give: never on macOS (Docker there has
// none), and on Linux and Windows when a throwaway container is given one. Checked once per image.
import { run } from "./exec";

const answers = new Map<string, Promise<boolean>>();

export function useGpu(image: string): Promise<boolean> {
  if (process.platform === "darwin") return Promise.resolve(false);
  let answer = answers.get(image);
  if (!answer) {
    answer = run("docker", ["run", "--rm", "--gpus", "all", "--entrypoint", "/bin/true", image], { timeout: 60_000 }).then(
      ({ code }) => code === 0,
    );
    answers.set(image, answer);
  }
  return answer;
}
