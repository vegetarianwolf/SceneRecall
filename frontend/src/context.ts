import { createContext, useContext } from 'react';
import type { Asset, Bootstrap, Job } from './types';

export interface AppContextValue {
  bootstrap: Bootstrap;
  assets: Asset[];
  jobs: Job[];
  refresh: () => Promise<void>;
  notify: (message: string, kind?: 'success' | 'error') => void;
  openImport: () => void;
  openJob: (asset: Asset) => void;
}
export const AppContext = createContext<AppContextValue | null>(null);
export function useApp(): AppContextValue {
  const app = useContext(AppContext);
  if (!app) throw new Error('App context unavailable');
  return app;
}
