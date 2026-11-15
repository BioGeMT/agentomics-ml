import type { Accounts, DesktopApi } from "@agentomics/contract";
import { AccountsCard } from "./AccountsCard";
import { AppearanceCard } from "./AppearanceCard";
import { ImagesCard } from "./ImagesCard";

/** Settings: appearance, model providers and the Docker image runs use. */
export function Settings({ api, accounts, onAccountsChange }: { api: DesktopApi; accounts: Accounts; onAccountsChange: () => void }) {
  return (
    <>
      <AppearanceCard api={api} />
      <AccountsCard api={api} accounts={accounts} onChange={onAccountsChange} />
      <ImagesCard api={api} />
    </>
  );
}
