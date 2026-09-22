import { useState, type FormEvent } from 'react';
import { ArrowRight, FileVideo, Subtitles } from 'lucide-react';
import { post, message, assetLink } from '../api';
import { useApp } from '../context';
import type { Asset, AssetKind } from '../types';
import { Alert, Field, Modal, Spinner } from './ui';

export default function ImportModal({ onClose }: { onClose: () => void }) {
  const { refresh, notify, bootstrap } = useApp();
  const [videoPath, setVideoPath] = useState('');
  const [title, setTitle] = useState('');
  const [titleEdited, setTitleEdited] = useState(false);
  const [kind, setKind] = useState<AssetKind>('movie');
  const [series, setSeries] = useState('');
  const [season, setSeason] = useState('');
  const [episode, setEpisode] = useState('');
  const [version, setVersion] = useState('');
  const [subtitleMode, setSubtitleMode] = useState<'external' | 'embedded'>('external');
  const [subtitlePath, setSubtitlePath] = useState('');
  const [offset, setOffset] = useState('0');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const cleanPath = (value: string) => value.trim().replace(/^['"]|['"]$/g, '');
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const asset = await post<Asset>('/assets', { video_path: cleanPath(videoPath), title: title.trim(), kind, series: series.trim() || '', season: season ? Number(season) : null, episode: episode ? Number(episode) : null, version: version.trim() || '', subtitle_mode: subtitleMode, subtitle_path: subtitleMode === 'external' ? cleanPath(subtitlePath) : null, subtitle_offset_ms: Math.round(Number(offset) * 1000) });
      await refresh(); notify('作品已加入本地资料库'); onClose(); window.location.hash = assetLink(asset.id).slice(1);
    } catch (error) { setError(message(error)); } finally { setBusy(false); }
  };

  return <Modal title="导入一部作品" subtitle="从本机视频开始，建立属于你的镜头资料。" onClose={onClose} wide><form onSubmit={submit}><div className="modal-body"><div className="file-entry"><FileVideo size={27} strokeWidth={1.5} /><div><strong>连接本地视频文件</strong><p>保留原文件位置，不会复制或上传整部视频。</p></div></div><Field label="视频绝对路径" hint="macOS：在 Finder 中选中文件，按 ⌥⌘C 复制路径后粘贴。"><input autoComplete="off" placeholder="/Users/你的名字/Movies/电影.mp4" value={videoPath} onChange={(e) => { setVideoPath(e.target.value); if (!titleEdited) setTitle(cleanPath(e.target.value).split('/').pop()?.replace(/\.[^.]+$/, '') || ''); }} required /></Field><div className="form-grid"><Field label="作品名称"><input placeholder="输入影片或单集名称" value={title} onChange={(e) => { setTitle(e.target.value); setTitleEdited(true); }} required /></Field><Field label="作品类型"><select value={kind} onChange={(e) => setKind(e.target.value as AssetKind)}><option value="movie">电影</option><option value="animation">动画</option><option value="series">剧集</option></select></Field></div>{kind !== 'movie' && <div className="form-grid three"><Field label="系列名称（可选）"><input value={series} onChange={(e) => setSeries(e.target.value)} placeholder="剧名 / 动画系列" /></Field><Field label="季（可选）"><input type="number" min="0" value={season} onChange={(e) => setSeason(e.target.value)} /></Field><Field label="集（可选）"><input type="number" min="0" value={episode} onChange={(e) => setEpisode(e.target.value)} /></Field></div>}<Field label="视频版本（可选）"><input value={version} onChange={(e) => setVersion(e.target.value)} placeholder="例如：院线版、导演剪辑版、1080p" /></Field><div className="field-label section-label">字幕来源</div><div className="choice-grid"><button type="button" className={`choice ${subtitleMode === 'external' ? 'selected' : ''}`} onClick={() => setSubtitleMode('external')}><Subtitles size={21} /><strong>外挂字幕文件</strong><span>导入 SRT / VTT / ASS，保留原文和时间轴</span></button><button type="button" className={`choice ${subtitleMode === 'embedded' ? 'selected' : ''}`} onClick={() => setSubtitleMode('embedded')}><FileVideo size={21} /><strong>画面内嵌字幕</strong><span>后续通过独立的视觉模型识别画面文字</span></button></div>{subtitleMode === 'external' ? <div className="form-grid subtitle-path"><Field label="字幕文件绝对路径 · 必填"><input required value={subtitlePath} onChange={(e) => setSubtitlePath(e.target.value)} placeholder="/Users/你的名字/Movies/电影.srt" /></Field><Field label="时间偏移（秒）" hint="正值表示延后"><input type="number" step="0.1" value={offset} onChange={(e) => setOffset(e.target.value)} required /></Field></div> : <Alert tone={bootstrap.settings.bindings.subtitle ? "info" : "warning"}>{bootstrap.settings.bindings.subtitle ? "导入后可单独启动字幕识别。模型只读取画面中可见的文字，音轨转写暂不支持。" : <>请先在<a href="#/settings" onClick={onClose}>模型与设置</a>中绑定字幕视觉模型，再导入内嵌字幕视频。</>}</Alert>}{error && <Alert tone="error">{error}</Alert>}</div><footer className="modal-footer"><span>导入时检查媒体与字幕格式</span><button type="submit" className="button primary" disabled={busy || (subtitleMode === 'embedded' && !bootstrap.settings.bindings.subtitle)}>{busy ? <Spinner label="正在校验文件" /> : <>加入资料库 <ArrowRight size={17} /></>}</button></footer></form></Modal>;
}
