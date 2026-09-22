import { useState, type FormEvent } from 'react';
import { ArrowUpRight, CheckCircle2, Clock3, Pause, Play, RefreshCw, RotateCcw, Workflow, X } from 'lucide-react';
import { assetLink, friendlyDate, message, post } from '../api';
import { useApp } from '../context';
import type { Job } from '../types';
import { Alert, Empty, Field, Modal, PageHeader, Spinner, Status } from './ui';

export default function Jobs() {
  const { jobs, assets, refresh, notify } = useApp();
  const [filter, setFilter] = useState('all');
  const [busy, setBusy] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [resuming, setResuming] = useState<{ job: Job; action: 'resume' | 'retry' } | null>(null);
  const active = ['running', 'queued', 'pending', 'pausing'];
  const change = async (job: Job, action: string, budget: Record<string, number | null> = {}) => {
    setBusy(job.id);
    try { await post(`/jobs/${encodeURIComponent(job.id)}/${action}`, budget); await refresh(); notify(({ pause: '已请求暂停，当前请求完成后生效', resume: '任务已恢复', cancel: '已请求取消，已完成的资料会保留', retry: '失败任务已重新排队' })[action as 'pause']); return true; }
    catch (error) { notify(message(error), 'error'); return false; } finally { setBusy(''); }
  };
  const filtered = jobs.filter((job) => filter === 'all' || (filter === 'active' ? active.includes(job.status) || job.status === 'paused' : filter === 'failed' ? ['failed', 'partial'].includes(job.status) : ['completed', 'complete'].includes(job.status)));
  const typeLabel = (type: string) => ({ analysis: '影视分析', analyze: '影视分析', proxy: '播放代理', reindex: '索引重建', index: '索引重建' })[type] || type;
  const stageLabel = (stage?: string) => ({ vision: '画面理解', subtitle: '字幕识别', embedding: '语义向量', probe: '媒体探测', shots: '镜头检测', segmentation: '镜头检测', index: '更新索引', queued: '等待处理' })[stage || ''] || stage || '准备中';
  return <><PageHeader eyebrow="WORK IN PROGRESS" title="任务队列" action={<button className="button secondary" disabled={refreshing} onClick={async () => { setRefreshing(true); try { await refresh(); } catch (error) { notify(message(error), 'error'); } finally { setRefreshing(false); } }}><RefreshCw size={16} className={refreshing ? 'spin' : ''} />刷新</button>}>分析在本机后台运行。暂停、恢复与重试，都保留已完成的资料。</PageHeader><div className="queue-stats"><div><Workflow size={21} /><strong>{jobs.filter((job) => active.includes(job.status)).length}</strong><span>进行中</span></div><div><CheckCircle2 size={21} /><strong>{jobs.filter((job) => ['completed', 'complete'].includes(job.status)).length}</strong><span>已完成</span></div><div><Clock3 size={21} /><strong>{jobs.filter((job) => job.status === 'paused').length}</strong><span>已暂停</span></div></div><div className="section-row"><div className="tabs">{[['all', '全部任务'], ['active', '进行中'], ['completed', '已完成'], ['failed', '失败']].map(([value, label]) => <button key={value} className={value === filter ? 'active' : ''} onClick={() => setFilter(value)}>{label}</button>)}</div><span className="helper">每 4 秒自动更新</span></div>{filtered.length === 0 ? <Empty icon={<Workflow size={29} />} title={jobs.length ? '这里暂时没有任务' : '还没有分析任务'} action={<a className="button secondary" href="#/library">返回资料库 <ArrowUpRight size={16} /></a>}>导入作品后，从作品详情页启动画面或字幕识别。</Empty> : <div className="job-list">{filtered.map((job) => {
    const asset = assets.find((asset) => asset.id === job.asset_id);
    const progress = job.total > 0 ? Math.min(100, Math.round(job.completed / job.total * 100)) : job.progress > 1 ? Math.min(100, job.progress) : Math.round((job.progress || 0) * 100);
    return <article className="job-card" key={job.id}><div className="job-header"><div className="job-icon"><Workflow size={21} /></div><div><h3>{asset ? <a href={assetLink(asset.id)}>{asset.title}</a> : typeLabel(job.type)}</h3><p>{typeLabel(job.type)} · {stageLabel(job.stage)} · {friendlyDate(job.created_at)}</p></div><Status status={job.status} /></div><div className="progress-label"><span>{job.message || stageLabel(job.stage)}</span><span className="mono">{job.total ? `${job.completed} / ${job.total}` : `${progress}%`}</span></div><div className="progress-track"><div style={{ width: `${progress}%` }} /></div>{job.error && <div className="job-error">{job.error}</div>}{job.cost_incomplete && <p className="helper">上次请求中断，费用记录不完整{job.uncertain_request_count ? `；${job.uncertain_request_count} 次请求的完成状态未知` : ''}。请以供应商账单为准。</p>}<div className="job-bottom"><div className="job-metrics"><span>API 请求 <strong>{job.request_count || 0}</strong></span><span>费用 <strong>{job.cost == null ? '未知' : `$${job.cost.toFixed(4)}`}</strong></span>{job.usage?.total_tokens != null && <span>Token <strong>{job.usage.total_tokens.toLocaleString()}</strong></span>}</div><div className="job-actions">{busy === job.id ? <Spinner label="正在处理" /> : <>{['running', 'queued', 'pending'].includes(job.status) && <button className="button small secondary" onClick={() => change(job, 'pause')}><Pause size={14} />暂停</button>}{['paused', 'interrupted'].includes(job.status) && <button className="button small primary" onClick={() => ['analysis', 'analyze'].includes(job.type) ? setResuming({ job, action: 'resume' }) : void change(job, 'resume')}><Play size={14} />恢复</button>}{['failed', 'partial', 'cancelled', 'canceled'].includes(job.status) && <button className="button small secondary" onClick={() => ['analysis', 'analyze'].includes(job.type) ? setResuming({ job, action: 'retry' }) : void change(job, 'retry')}><RotateCcw size={14} />重试</button>}{[...active, 'paused'].includes(job.status) && <button className="button small ghost" onClick={() => change(job, 'cancel')}><X size={14} />取消</button>}</>}</div></div></article>;
  })}</div>}{resuming && <ResumeModal job={resuming.job} action={resuming.action} onClose={() => setResuming(null)} onResume={async (budget) => { const ok = await change(resuming.job, resuming.action, budget); if (ok) setResuming(null); return ok; }} />}</>;
}


function ResumeModal({ job, action, onClose, onResume }: { job: Job; action: 'resume' | 'retry'; onClose: () => void; onResume: (budget: Record<string, number | null>) => Promise<boolean> }) {
  const [maxRequests, setMaxRequests] = useState(job.budget?.max_requests || Math.max(1000, (job.request_count || 0) + 100));
  const [maxCost, setMaxCost] = useState(job.budget?.max_cost?.toString() || '');
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true);
    await onResume({ max_requests: maxRequests, max_cost: maxCost === '' ? null : Number(maxCost) });
    setBusy(false);
  };
  return <Modal title={action === 'retry' ? '重试未完成的任务' : '恢复分析任务'} subtitle="复用已成功的结果，可在继续之前调整预算。" onClose={onClose}><form onSubmit={submit}><div className="modal-body"><Alert tone="info">预算是任务的累计上限。目前已请求 {job.request_count || 0} 次；若因预算暂停，请提高上限后继续。</Alert><Field label="累计请求次数上限"><input type="number" min={(job.request_count || 0) + 1} max="100000" value={maxRequests} onChange={(event) => setMaxRequests(Number(event.target.value))} required /></Field><Field label="累计费用上限（USD）" hint="留空会移除原有费用限制；请求次数限制继续生效。"><input type="number" min="0.01" step="0.01" value={maxCost} onChange={(event) => setMaxCost(event.target.value)} placeholder="未设置" /></Field>{job.cost_incomplete && <Alert>中断导致费用无法完整核对。若继续保留费用上限，任务可能再次暂停；可以移除费用限制，并用请求次数限制控制用量。</Alert>}</div><footer className="modal-footer"><span>成功结果与人工修改会保留</span><button type="submit" className="button primary" disabled={busy}>{busy ? <Spinner label="提交中" /> : <><Play size={15} />继续任务</>}</button></footer></form></Modal>;
}
