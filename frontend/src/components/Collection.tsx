import { useEffect, useState } from 'react';
import { Bookmark, Download } from 'lucide-react';
import { api, message } from '../api';
import type { SearchResult } from '../types';
import { Alert, Empty, PageHeader, Spinner } from './ui';
import { ResultCard } from './Search';

export default function Collection() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const load = () => api<SearchResult[]>('/collections').then(setResults).catch((error) => setError(message(error))).finally(() => setLoading(false));
  useEffect(() => { void load(); }, []);
  return <><PageHeader eyebrow="COLLECT THE DETAILS" title="片段收藏" action={<div className="export-buttons"><a className={`button secondary ${results.length ? '' : 'disabled'}`} href="/api/export/citations?format=json" download>JSON</a><a className={`button primary ${results.length ? '' : 'disabled'}`} href="/api/export/citations?format=markdown" download><Download size={17} />导出引用</a></div>}>把有用的画面和台词留在手边，为下一篇影评积累证据。</PageHeader>{error && <Alert tone="error">{error}</Alert>}{loading ? <Spinner /> : results.length === 0 ? <Empty icon={<Bookmark size={29} />} title="还没有收藏的片段" action={<a href="#/search" className="button secondary">去检索片段</a>}>在搜索结果或作品详情中点击收藏。你的笔记与影片时间码会一起保留。</Empty> : <><div className="section-row"><h3>{results.length} 个收藏片段</h3><span className="helper">引用包含作品、时间范围与记录来源</span></div><div className="results-list">{results.map((result) => <ResultCard key={`${result.asset_id}-${result.id}`} result={result} saved onSaved={() => { void load(); }} />)}</div></>}</>;
}
