import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { AppContext, type AppContextValue } from '../src/context';
import { DEFAULTS, type Asset, type Bootstrap, type Job, type Profile } from '../src/types';
import ImportModal from '../src/components/ImportModal';
import JobModal from '../src/components/JobModal';
import SearchPage from '../src/components/Search';
import Settings from '../src/components/Settings';
import Jobs from '../src/components/Jobs';
import App from '../src/App';

const asset: Asset = { id: 'synthetic-asset', work_id: 'synthetic-work', title: '合成测试片段', kind: 'movie', duration_ms: 12000, width: 640, height: 360, subtitle_mode: 'external', source_available: true, source_changed: false, created_at: '2026-01-01T00:00:00Z', status: 'ready' };
const profile: Profile = { id: 'vision-test', name: '测试视觉接口', base_url: 'http://127.0.0.1:9000/v1', model: 'local-test-model', capabilities: ['vision'], secret_mode: 'session', has_key: true };
const bootstrap: Bootstrap = { library_path: '/tmp/scenerecall-ui-tests', settings: { bindings: {}, defaults: DEFAULTS }, provider_profiles: [], stats: { assets: 1, records: 1, favorites: 0 }, tools: { ffmpeg: true, ffprobe: true }, version: '0.1.0' };

function appContext(overrides: Partial<AppContextValue> = {}): AppContextValue {
  return { bootstrap, assets: [asset], jobs: [], refresh: vi.fn().mockResolvedValue(undefined), notify: vi.fn(), openImport: vi.fn(), openJob: vi.fn(), ...overrides };
}
function renderApp(ui: ReactNode, context: AppContextValue = appContext()) {
  return render(<AppContext.Provider value={context}>{ui}</AppContext.Provider>);
}
function mockFetch(handler: (url: string, options?: RequestInit) => unknown | Promise<unknown>) {
  const mock = vi.fn(async (url: string, options?: RequestInit) => new Response(JSON.stringify(await handler(url, options)), { status: 200, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', mock);
  return mock;
}
afterEach(() => {
  cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals();
  window.history.replaceState({}, '', '/');
});

describe('Import and model prerequisites', () => {
  it('requires a subtitle path before sending an external-subtitle import', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => asset);
    const onClose = vi.fn();
    renderApp(<ImportModal onClose={onClose} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/synthetic.mp4');
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    expect(fetch).not.toHaveBeenCalled();
    expect((screen.getByLabelText(/字幕文件绝对路径/) as HTMLInputElement).validity.valueMissing).toBe(true);
    await user.type(screen.getByLabelText(/字幕文件绝对路径/), '/tmp/synthetic.srt');
    await user.click(screen.getByRole('button', { name: /加入资料库/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toMatchObject({ video_path: '/tmp/synthetic.mp4', subtitle_path: '/tmp/synthetic.srt', subtitle_mode: 'external', series: '', title: 'synthetic' });
    expect(window.location.hash).toBe('#/asset/synthetic-asset');
  });

  it('blocks embedded import until a subtitle model has been assigned', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => asset);
    renderApp(<ImportModal onClose={vi.fn()} />);
    await user.type(screen.getByLabelText(/视频绝对路径/), '/tmp/synthetic.mp4');
    await user.click(screen.getByRole('button', { name: /画面内嵌字幕/ }));
    expect((screen.getByRole('button', { name: /加入资料库/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/中绑定字幕视觉模型/)).toBeTruthy();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('keeps an existing API key input empty and omits it on unchanged profile updates', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => profile);
    renderApp(<Settings />, appContext({ bootstrap: { ...bootstrap, provider_profiles: [profile] } }));
    await user.click(screen.getByRole('button', { name: /编辑 测试视觉接口/ }));
    const dialog = screen.getByRole('dialog');
    const keyInput = within(dialog).getByLabelText(/API Key/) as HTMLInputElement;
    expect(keyInput.type).toBe('password');
    expect(keyInput.value).toBe('');
    expect(window.localStorage.length).toBe(0);
    await user.click(within(dialog).getByRole('button', { name: /保存连接/ }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    const payload = JSON.parse(fetch.mock.calls[0][1]!.body as string);
    expect(payload.id).toBe(profile.id);
    expect(payload).not.toHaveProperty('api_key');
    expect(payload.capabilities).toEqual(['vision']);
  });
});

describe('Analysis budgeting and independent stages', () => {
  const estimate = { duration_ms: 12000, estimated_frames: 8, estimated_requests: 2, cost_estimate: null, warnings: [] };
  it('does not start a job with a missing model binding, even after a returned estimate', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => estimate);
    renderApp(<JobModal asset={asset} onClose={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /估算用量/ }));
    await screen.findByText(/重新估算/);
    expect((screen.getByRole('button', { name: /开始分析/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetch.mock.calls.map((call) => call[0])).toEqual(['/api/jobs/estimate']);
  });

  it('requires re-estimation after a sampling change and submits only the selected stage', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url.endsWith('/estimate') ? estimate : { id: 'synthetic-job', status: 'queued' });
    const onClose = vi.fn();
    const context = appContext({ bootstrap: { ...bootstrap, provider_profiles: [profile], settings: { ...bootstrap.settings, bindings: { vision: profile.id } } } });
    renderApp(<JobModal asset={asset} onClose={onClose} />, context);
    const startButton = screen.getByRole('button', { name: /开始分析/ }) as HTMLButtonElement;
    expect(startButton.disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: /估算用量/ }));
    await waitFor(() => expect(startButton.disabled).toBe(false));
    const requests = screen.getByLabelText(/最多 API 请求数/);
    await user.clear(requests); await user.type(requests, '50');
    expect(startButton.disabled).toBe(true);
    await user.click(screen.getByRole('button', { name: /估算用量/ }));
    await waitFor(() => expect(startButton.disabled).toBe(false));
    await user.click(startButton);
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    const jobCall = fetch.mock.calls.find((call) => call[0] === '/api/jobs');
    expect(jobCall).toBeTruthy();
    expect(JSON.parse(jobCall![1]!.body as string)).toMatchObject({ asset_id: asset.id, stages: ['vision'], max_requests: 50, max_cost: null });
  });

  it('allows retrying partial jobs with revised cumulative budgets', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch(() => ({}));
    const job: Job = { id: 'partial-job', type: 'analysis', status: 'partial', progress: 0.5, completed: 1, total: 2, asset_id: asset.id, request_count: 2, created_at: asset.created_at, budget: { max_requests: 2, max_cost: 1 } };
    renderApp(<Jobs />, appContext({ jobs: [job] }));
    expect(screen.getByText('部分完成')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /重试/ }));
    const dialog = screen.getByRole('dialog');
    const requests = within(dialog).getByLabelText(/累计请求次数上限/);
    await user.clear(requests); await user.type(requests, '20');
    expect((within(dialog).getByLabelText(/累计费用上限/) as HTMLInputElement).value).toBe('1');
    await user.clear(within(dialog).getByLabelText(/累计费用上限/));
    await user.click(within(dialog).getByRole('button', { name: /继续任务/ }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(fetch.mock.calls[0][0]).toBe('/api/jobs/partial-job/retry');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ max_requests: 20, max_cost: null });
  });
});

describe('Evidence retrieval and session authentication', () => {
  it('shows degraded search with playable evidence and saves its source record', async () => {
    const user = userEvent.setup();
    const fetch = mockFetch((url) => url === '/api/search' ? { results: [{ id: 'cue-result', record_id: 'cue-source', asset_id: asset.id, title: asset.title, kind: 'subtitle', start_ms: 3500, end_ms: 5700, text: '下一站，我们再见。', review_status: 'reviewed', match_type: 'full', match_reason: '原字幕逐字匹配' }], degraded: ['语义索引尚未构建，已使用字面检索'] } : {});
    renderApp(<SearchPage />);
    await user.type(screen.getByRole('textbox', { name: /用自然语言搜索/ }), '下一站');
    await user.click(screen.getByRole('button', { name: /^检索/ }));
    await screen.findByText('下一站，我们再见。');
    expect(screen.getByText(/语义索引尚未构建/)).toBeTruthy();
    expect(screen.getByRole('link', { name: /查看原片/ }).getAttribute('href')).toBe('#/asset/synthetic-asset?t=3500');
    await user.click(screen.getByRole('button', { name: /收藏片段/ }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(fetch.mock.calls[1][0]).toBe('/api/assets/synthetic-asset/annotations');
    expect(JSON.parse(fetch.mock.calls[1][1]!.body as string)).toEqual({ record_id: 'cue-source', favorite: true });
    expect(fetch.mock.calls.map((call) => call[0]).some((url) => /jobs|analy|recogn/.test(url))).toBe(false);
  });

  it('exchanges a startup token before reading the library and removes it from the URL', async () => {
    window.history.replaceState({}, '', '/?token=synthetic-local-token#/library');
    const fetch = mockFetch((url) => url === '/api/session' ? { ok: true } : url === '/api/bootstrap' ? bootstrap : []);
    render(<App />);
    await screen.findByRole('heading', { name: '我的资料库' });
    expect(window.location.search).toBe('');
    expect(fetch.mock.calls[0][0]).toBe('/api/session');
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string)).toEqual({ token: 'synthetic-local-token' });
    expect(window.localStorage.length).toBe(0);
    expect(screen.queryByText('synthetic-local-token')).toBeNull();
  });
});
