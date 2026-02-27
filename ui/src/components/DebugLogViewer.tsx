/**
 * Debug Log Viewer Component
 *
 * Collapsible panel at the bottom of the screen showing real-time
 * agent output (tool calls, results, steps). Similar to browser DevTools.
 * Features a resizable height via drag handle and tabs for different log sources.
 */

import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { ChevronUp, ChevronDown, Trash2, Terminal as TerminalIcon, GripHorizontal, Cpu, Server, Zap } from 'lucide-react'
import { Terminal } from './Terminal'
import { TerminalTabs } from './TerminalTabs'
import { listTerminals, createTerminal, renameTerminal, deleteTerminal } from '@/lib/api'
import type { TerminalInfo } from '@/lib/types'
import { AGENT_MASCOTS } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

const MIN_HEIGHT = 150
const MAX_HEIGHT = 600
const DEFAULT_HEIGHT = 288
const STORAGE_KEY = 'debug-panel-height'
const TAB_STORAGE_KEY = 'debug-panel-tab'

type TabType = 'agent' | 'devserver' | 'terminal' | 'apicalls'

type LogEntry = { line: string; timestamp: string; featureId?: number; agentIndex?: number }

interface DebugLogViewerProps {
  logs: LogEntry[]
  devLogs: Array<{ line: string; timestamp: string }>
  isOpen: boolean
  onToggle: () => void
  onClear: () => void
  onClearDevLogs: () => void
  onHeightChange?: (height: number) => void
  projectName: string
  activeTab?: TabType
  onTabChange?: (tab: TabType) => void
}

type LogLevel = 'error' | 'warn' | 'debug' | 'info'

// ─── API Call Parsing ────────────────────────────────────────────────────────

type ApiCallType = 'tool' | 'usage' | 'rate_limit' | 'error'

interface ApiCallEntry {
  id: string
  timestamp: string
  featureId?: number
  agentIndex?: number
  callType: ApiCallType
  tool: string    // e.g. "Read", "Write", "Bash", "Usage", "RateLimit"
  detail: string  // Cleaned-up content (stripped of feature/tool prefix)
  raw: string     // Full raw log line
}

const TOOL_PATTERN = /\[Tool:\s*(\w+)\]/i
const COST_PATTERN = /(?:total cost|cost:|api usage|session cost)\s*[\$€£]?[\d.]+/i
const RATE_LIMIT_PATTERN = /rate.?limit|too many requests|429|retry.?after/i
const API_ERROR_PATTERN = /api.?error|request.?failed|connection.?error|ssl.?error|timeout.*api/i

function parseApiCalls(logs: LogEntry[]): ApiCallEntry[] {
  const entries: ApiCallEntry[] = []

  for (let i = 0; i < logs.length; i++) {
    const log = logs[i]
    const line = log.line

    const toolMatch = TOOL_PATTERN.exec(line)
    if (toolMatch) {
      const detail = line
        .replace(/^\[Feature #\d+\]\s*/, '')
        .replace(/\[Tool:\s*\w+\]\s*/, '')
        .trim()
      entries.push({
        id: `${i}`,
        timestamp: log.timestamp,
        featureId: log.featureId,
        agentIndex: log.agentIndex,
        callType: 'tool',
        tool: toolMatch[1],
        detail,
        raw: line,
      })
      continue
    }

    if (RATE_LIMIT_PATTERN.test(line)) {
      entries.push({
        id: `${i}`,
        timestamp: log.timestamp,
        featureId: log.featureId,
        agentIndex: log.agentIndex,
        callType: 'rate_limit',
        tool: 'RateLimit',
        detail: line.replace(/^\[Feature #\d+\]\s*/, '').trim(),
        raw: line,
      })
      continue
    }

    if (API_ERROR_PATTERN.test(line)) {
      entries.push({
        id: `${i}`,
        timestamp: log.timestamp,
        featureId: log.featureId,
        agentIndex: log.agentIndex,
        callType: 'error',
        tool: 'APIError',
        detail: line.replace(/^\[Feature #\d+\]\s*/, '').trim(),
        raw: line,
      })
      continue
    }

    if (COST_PATTERN.test(line)) {
      entries.push({
        id: `${i}`,
        timestamp: log.timestamp,
        featureId: log.featureId,
        agentIndex: log.agentIndex,
        callType: 'usage',
        tool: 'Usage',
        detail: line.replace(/^\[Feature #\d+\]\s*/, '').trim(),
        raw: line,
      })
    }
  }

  return entries
}

// ─── Tool colour + label helpers ────────────────────────────────────────────

function getToolColor(tool: string): string {
  switch (tool.toLowerCase()) {
    case 'read': return 'text-blue-400'
    case 'write':
    case 'edit':
    case 'notebookedit': return 'text-green-400'
    case 'bash': return 'text-yellow-400'
    case 'glob':
    case 'grep': return 'text-purple-400'
    case 'task': return 'text-cyan-400'
    case 'webfetch':
    case 'websearch': return 'text-sky-400'
    case 'usage': return 'text-teal-400'
    case 'ratelimit': return 'text-orange-500'
    case 'apierror': return 'text-red-500'
    default: return 'text-muted-foreground'
  }
}

function getToolBgColor(tool: string): string {
  switch (tool.toLowerCase()) {
    case 'read': return 'bg-blue-950/30 border-blue-800/40'
    case 'write':
    case 'edit':
    case 'notebookedit': return 'bg-green-950/30 border-green-800/40'
    case 'bash': return 'bg-yellow-950/30 border-yellow-800/40'
    case 'glob':
    case 'grep': return 'bg-purple-950/30 border-purple-800/40'
    case 'task': return 'bg-cyan-950/30 border-cyan-800/40'
    case 'usage': return 'bg-teal-950/30 border-teal-800/40'
    case 'ratelimit': return 'bg-orange-950/30 border-orange-800/40'
    case 'apierror': return 'bg-red-950/30 border-red-800/40'
    default: return 'bg-muted/30 border-border'
  }
}

// ─── ApiCallRow sub-component ────────────────────────────────────────────────

function ApiCallRow({
  entry,
  formatTimestamp,
}: {
  entry: ApiCallEntry
  formatTimestamp: (ts: string) => string
}) {
  const [expanded, setExpanded] = useState(false)
  const colorClass = getToolColor(entry.tool)
  const bgClass = getToolBgColor(entry.tool)
  const agentName =
    entry.agentIndex !== undefined
      ? AGENT_MASCOTS[entry.agentIndex % AGENT_MASCOTS.length]
      : null

  return (
    <div
      className={`border rounded px-2 py-1 cursor-pointer hover:brightness-110 transition-all ${bgClass}`}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Overview row */}
      <div className="flex gap-2 items-center text-xs font-mono min-w-0">
        <span className="text-muted-foreground shrink-0 tabular-nums">
          {formatTimestamp(entry.timestamp)}
        </span>

        <span className={`${colorClass} font-bold w-20 shrink-0 truncate`} title={entry.tool}>
          {entry.tool}
        </span>

        {entry.featureId !== undefined && (
          <span className="text-muted-foreground shrink-0">#{entry.featureId}</span>
        )}

        {agentName && (
          <span className="text-cyan-500 shrink-0">{agentName}</span>
        )}

        <span className="text-foreground/80 truncate flex-1 min-w-0">
          {entry.detail || <span className="italic text-muted-foreground">—</span>}
        </span>

        <ChevronDown
          size={12}
          className={`shrink-0 text-muted-foreground transition-transform duration-150 ${expanded ? 'rotate-180' : ''}`}
        />
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="mt-1.5 pt-1.5 border-t border-border/50 text-xs font-mono text-muted-foreground break-all whitespace-pre-wrap">
          {entry.raw}
        </div>
      )}
    </div>
  )
}

// ─── Main component ──────────────────────────────────────────────────────────

export function DebugLogViewer({
  logs,
  devLogs,
  isOpen,
  onToggle,
  onClear,
  onClearDevLogs,
  onHeightChange,
  projectName,
  activeTab: controlledActiveTab,
  onTabChange,
}: DebugLogViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const devScrollRef = useRef<HTMLDivElement>(null)
  const apiScrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [devAutoScroll, setDevAutoScroll] = useState(true)
  const [apiAutoScroll, setApiAutoScroll] = useState(true)
  const [isResizing, setIsResizing] = useState(false)
  const [panelHeight, setPanelHeight] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? Math.min(Math.max(parseInt(saved, 10), MIN_HEIGHT), MAX_HEIGHT) : DEFAULT_HEIGHT
  })
  const [internalActiveTab, setInternalActiveTab] = useState<TabType>(() => {
    const saved = localStorage.getItem(TAB_STORAGE_KEY)
    return (saved as TabType) || 'agent'
  })

  // Terminal management state
  const [terminals, setTerminals] = useState<TerminalInfo[]>([])
  const [activeTerminalId, setActiveTerminalId] = useState<string | null>(null)
  const [isLoadingTerminals, setIsLoadingTerminals] = useState(false)

  const activeTab = controlledActiveTab ?? internalActiveTab
  const setActiveTab = (tab: TabType) => {
    setInternalActiveTab(tab)
    localStorage.setItem(TAB_STORAGE_KEY, tab)
    onTabChange?.(tab)
  }

  // Parsed API call entries (derived from logs)
  const apiCallEntries = useMemo(() => parseApiCalls(logs), [logs])

  // Fetch terminals for the project
  const fetchTerminals = useCallback(async () => {
    if (!projectName) return

    setIsLoadingTerminals(true)
    try {
      const terminalList = await listTerminals(projectName)
      setTerminals(terminalList)

      if (terminalList.length > 0) {
        if (!activeTerminalId || !terminalList.find((t) => t.id === activeTerminalId)) {
          setActiveTerminalId(terminalList[0].id)
        }
      }
    } catch (err) {
      console.error('Failed to fetch terminals:', err)
    } finally {
      setIsLoadingTerminals(false)
    }
  }, [projectName, activeTerminalId])

  const handleCreateTerminal = useCallback(async () => {
    if (!projectName) return
    try {
      const newTerminal = await createTerminal(projectName)
      setTerminals((prev) => [...prev, newTerminal])
      setActiveTerminalId(newTerminal.id)
    } catch (err) {
      console.error('Failed to create terminal:', err)
    }
  }, [projectName])

  const handleRenameTerminal = useCallback(
    async (terminalId: string, newName: string) => {
      if (!projectName) return
      try {
        const updated = await renameTerminal(projectName, terminalId, newName)
        setTerminals((prev) => prev.map((t) => (t.id === terminalId ? updated : t)))
      } catch (err) {
        console.error('Failed to rename terminal:', err)
      }
    },
    [projectName]
  )

  const handleCloseTerminal = useCallback(
    async (terminalId: string) => {
      if (!projectName || terminals.length <= 1) return
      try {
        await deleteTerminal(projectName, terminalId)
        setTerminals((prev) => prev.filter((t) => t.id !== terminalId))
        if (activeTerminalId === terminalId) {
          const remaining = terminals.filter((t) => t.id !== terminalId)
          if (remaining.length > 0) setActiveTerminalId(remaining[0].id)
        }
      } catch (err) {
        console.error('Failed to close terminal:', err)
      }
    },
    [projectName, terminals, activeTerminalId]
  )

  useEffect(() => {
    if (projectName) {
      fetchTerminals()
    } else {
      setTerminals([])
      setActiveTerminalId(null)
    }
  }, [projectName]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll when new logs arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current && isOpen && activeTab === 'agent') {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs, autoScroll, isOpen, activeTab])

  useEffect(() => {
    if (devAutoScroll && devScrollRef.current && isOpen && activeTab === 'devserver') {
      devScrollRef.current.scrollTop = devScrollRef.current.scrollHeight
    }
  }, [devLogs, devAutoScroll, isOpen, activeTab])

  useEffect(() => {
    if (apiAutoScroll && apiScrollRef.current && isOpen && activeTab === 'apicalls') {
      apiScrollRef.current.scrollTop = apiScrollRef.current.scrollHeight
    }
  }, [apiCallEntries, apiAutoScroll, isOpen, activeTab])

  useEffect(() => {
    if (onHeightChange && isOpen) onHeightChange(panelHeight)
  }, [panelHeight, isOpen, onHeightChange])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    const newHeight = window.innerHeight - e.clientY
    setPanelHeight(Math.min(Math.max(newHeight, MIN_HEIGHT), MAX_HEIGHT))
  }, [])

  const handleMouseUp = useCallback(() => {
    setIsResizing(false)
    localStorage.setItem(STORAGE_KEY, panelHeight.toString())
  }, [panelHeight])

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'ns-resize'
      document.body.style.userSelect = 'none'
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing, handleMouseMove, handleMouseUp])

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsResizing(true)
  }

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    setAutoScroll(el.scrollHeight - el.scrollTop <= el.clientHeight + 50)
  }

  const handleDevScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    setDevAutoScroll(el.scrollHeight - el.scrollTop <= el.clientHeight + 50)
  }

  const handleApiScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    setApiAutoScroll(el.scrollHeight - el.scrollTop <= el.clientHeight + 50)
  }

  const handleClear = () => {
    if (activeTab === 'agent') {
      onClear()
    } else if (activeTab === 'devserver') {
      onClearDevLogs()
    } else if (activeTab === 'apicalls') {
      // API calls are derived from logs; clearing logs clears them too
      onClear()
    }
  }

  const getCurrentLogCount = () => {
    if (activeTab === 'agent') return logs.length
    if (activeTab === 'devserver') return devLogs.length
    if (activeTab === 'apicalls') return apiCallEntries.length
    return 0
  }

  const isAutoScrollPaused = () => {
    if (activeTab === 'agent') return !autoScroll
    if (activeTab === 'devserver') return !devAutoScroll
    if (activeTab === 'apicalls') return !apiAutoScroll
    return false
  }

  const getLogLevel = (line: string): LogLevel => {
    const lowerLine = line.toLowerCase()
    if (lowerLine.includes('error') || lowerLine.includes('exception') || lowerLine.includes('traceback')) return 'error'
    if (lowerLine.includes('warn') || lowerLine.includes('warning')) return 'warn'
    if (lowerLine.includes('debug')) return 'debug'
    return 'info'
  }

  const getLogColor = (level: LogLevel): string => {
    switch (level) {
      case 'error': return 'text-red-500'
      case 'warn': return 'text-yellow-500'
      case 'debug': return 'text-blue-400'
      default: return 'text-foreground'
    }
  }

  const formatTimestamp = (timestamp: string): string => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    } catch {
      return ''
    }
  }

  return (
    <div
      className={`fixed bottom-0 left-0 right-0 z-40 ${isResizing ? '' : 'transition-all duration-200'}`}
      style={{ height: isOpen ? panelHeight : 40 }}
    >
      {/* Resize handle */}
      {isOpen && (
        <div
          className="absolute top-0 left-0 right-0 h-2 cursor-ns-resize group flex items-center justify-center -translate-y-1/2 z-50"
          onMouseDown={handleResizeStart}
        >
          <div className="w-16 h-1.5 bg-border rounded-full group-hover:bg-muted-foreground transition-colors flex items-center justify-center">
            <GripHorizontal size={12} className="text-muted-foreground group-hover:text-foreground" />
          </div>
        </div>
      )}

      {/* Header bar */}
      <div className="flex items-center justify-between h-10 px-4 bg-muted border-t border-border">
        <div className="flex items-center gap-2">
          <button
            onClick={onToggle}
            className="flex items-center gap-2 hover:bg-accent px-2 py-1 rounded transition-colors cursor-pointer"
          >
            <TerminalIcon size={16} className="text-green-500" />
            <span className="font-mono text-sm text-foreground font-bold">Debug</span>
            <Badge variant="secondary" className="text-xs font-mono" title="Toggle debug panel">D</Badge>
          </button>

          {/* Tabs */}
          {isOpen && (
            <div className="flex items-center gap-1 ml-4">
              <Button
                variant={activeTab === 'agent' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={(e: React.MouseEvent) => { e.stopPropagation(); setActiveTab('agent') }}
                className="h-7 text-xs font-mono gap-1.5"
              >
                <Cpu size={12} />
                Agent
                {logs.length > 0 && (
                  <Badge variant="default" className="h-4 px-1.5 text-[10px]">{logs.length}</Badge>
                )}
              </Button>

              <Button
                variant={activeTab === 'apicalls' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={(e: React.MouseEvent) => { e.stopPropagation(); setActiveTab('apicalls') }}
                className="h-7 text-xs font-mono gap-1.5"
              >
                <Zap size={12} />
                API Calls
                {apiCallEntries.length > 0 && (
                  <Badge variant="default" className="h-4 px-1.5 text-[10px]">{apiCallEntries.length}</Badge>
                )}
              </Button>

              <Button
                variant={activeTab === 'devserver' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={(e: React.MouseEvent) => { e.stopPropagation(); setActiveTab('devserver') }}
                className="h-7 text-xs font-mono gap-1.5"
              >
                <Server size={12} />
                Dev Server
                {devLogs.length > 0 && (
                  <Badge variant="default" className="h-4 px-1.5 text-[10px]">{devLogs.length}</Badge>
                )}
              </Button>

              <Button
                variant={activeTab === 'terminal' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={(e: React.MouseEvent) => { e.stopPropagation(); setActiveTab('terminal') }}
                className="h-7 text-xs font-mono gap-1.5"
              >
                <TerminalIcon size={12} />
                Terminal
                <Badge variant="outline" className="h-4 px-1.5 text-[10px]" title="Toggle terminal">T</Badge>
              </Button>
            </div>
          )}

          {/* Log count + pause indicator */}
          {isOpen && activeTab !== 'terminal' && (
            <>
              {getCurrentLogCount() > 0 && (
                <Badge variant="secondary" className="ml-2 font-mono">{getCurrentLogCount()}</Badge>
              )}
              {isAutoScrollPaused() && (
                <Badge variant="default" className="bg-yellow-500 text-yellow-950">Paused</Badge>
              )}
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          {isOpen && activeTab !== 'terminal' && (
            <Button
              variant="ghost"
              size="icon"
              onClick={(e: React.MouseEvent) => { e.stopPropagation(); handleClear() }}
              className="h-7 w-7"
              title="Clear logs"
            >
              <Trash2 size={14} className="text-muted-foreground" />
            </Button>
          )}
          <div className="p-1">
            {isOpen
              ? <ChevronDown size={16} className="text-muted-foreground" />
              : <ChevronUp size={16} className="text-muted-foreground" />
            }
          </div>
        </div>
      </div>

      {/* Content area */}
      {isOpen && (
        <div className="h-[calc(100%-2.5rem)] bg-card">

          {/* Agent Logs Tab */}
          {activeTab === 'agent' && (
            <div ref={scrollRef} onScroll={handleScroll} className="h-full overflow-y-auto p-2 font-mono text-sm">
              {logs.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  No logs yet. Start the agent to see output.
                </div>
              ) : (
                <div className="space-y-0.5">
                  {logs.map((log, index) => {
                    const level = getLogLevel(log.line)
                    const colorClass = getLogColor(level)
                    return (
                      <div key={`${log.timestamp}-${index}`} className="flex gap-2 hover:bg-muted px-1 py-0.5 rounded">
                        <span className="text-muted-foreground select-none shrink-0">{formatTimestamp(log.timestamp)}</span>
                        <span className={`${colorClass} whitespace-pre-wrap break-all`}>{log.line}</span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {/* API Calls Tab */}
          {activeTab === 'apicalls' && (
            <div ref={apiScrollRef} onScroll={handleApiScroll} className="h-full overflow-y-auto p-2">
              {apiCallEntries.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground font-mono text-sm">
                  <Zap size={20} className="opacity-30" />
                  <span>No API calls detected yet.</span>
                  <span className="text-xs opacity-60">Tool calls and cost events will appear here when the agent runs.</span>
                </div>
              ) : (
                <div className="space-y-1">
                  {/* Summary bar */}
                  <div className="flex items-center gap-3 px-1 pb-1 mb-1 border-b border-border text-xs text-muted-foreground font-mono">
                    <span>{apiCallEntries.filter(e => e.callType === 'tool').length} tool calls</span>
                    {apiCallEntries.filter(e => e.callType === 'rate_limit').length > 0 && (
                      <span className="text-orange-400">
                        {apiCallEntries.filter(e => e.callType === 'rate_limit').length} rate limit(s)
                      </span>
                    )}
                    {apiCallEntries.filter(e => e.callType === 'error').length > 0 && (
                      <span className="text-red-400">
                        {apiCallEntries.filter(e => e.callType === 'error').length} error(s)
                      </span>
                    )}
                    {apiCallEntries.filter(e => e.callType === 'usage').length > 0 && (
                      <span className="text-teal-400">
                        {apiCallEntries.filter(e => e.callType === 'usage').length} usage report(s)
                      </span>
                    )}
                  </div>

                  {apiCallEntries.map((entry) => (
                    <ApiCallRow key={entry.id} entry={entry} formatTimestamp={formatTimestamp} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Dev Server Logs Tab */}
          {activeTab === 'devserver' && (
            <div ref={devScrollRef} onScroll={handleDevScroll} className="h-full overflow-y-auto p-2 font-mono text-sm">
              {devLogs.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  No dev server logs yet.
                </div>
              ) : (
                <div className="space-y-0.5">
                  {devLogs.map((log, index) => {
                    const level = getLogLevel(log.line)
                    const colorClass = getLogColor(level)
                    return (
                      <div key={`${log.timestamp}-${index}`} className="flex gap-2 hover:bg-muted px-1 py-0.5 rounded">
                        <span className="text-muted-foreground select-none shrink-0">{formatTimestamp(log.timestamp)}</span>
                        <span className={`${colorClass} whitespace-pre-wrap break-all`}>{log.line}</span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {/* Terminal Tab */}
          {activeTab === 'terminal' && (
            <div className="h-full flex flex-col">
              {terminals.length > 0 && (
                <TerminalTabs
                  terminals={terminals}
                  activeTerminalId={activeTerminalId}
                  onSelect={setActiveTerminalId}
                  onCreate={handleCreateTerminal}
                  onRename={handleRenameTerminal}
                  onClose={handleCloseTerminal}
                />
              )}

              <div className="flex-1 min-h-0 relative">
                {isLoadingTerminals ? (
                  <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-sm">
                    Loading terminals...
                  </div>
                ) : terminals.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-sm">
                    No terminal available
                  </div>
                ) : (
                  terminals.map((terminal) => {
                    const isActiveTerminal = terminal.id === activeTerminalId
                    return (
                      <div
                        key={terminal.id}
                        className="absolute inset-0"
                        style={{
                          zIndex: isActiveTerminal ? 10 : 1,
                          transform: isActiveTerminal ? 'none' : 'translateX(-200%)',
                          pointerEvents: isActiveTerminal ? 'auto' : 'none',
                        }}
                      >
                        <Terminal
                          projectName={projectName}
                          terminalId={terminal.id}
                          isActive={activeTab === 'terminal' && isActiveTerminal}
                        />
                      </div>
                    )
                  })
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Export the TabType for use in parent components
export type { TabType }
