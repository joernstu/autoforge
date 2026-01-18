import { MessageCircle, ScrollText, X, Copy, Check, Code, FlaskConical } from 'lucide-react'
import { useState } from 'react'
import { createPortal } from 'react-dom'
import { AgentAvatar } from './AgentAvatar'
import type { ActiveAgent, AgentLogEntry, AgentType } from '../lib/types'

interface AgentCardProps {
  agent: ActiveAgent
  onShowLogs?: (agentIndex: number) => void
}

// Get a friendly state description
function getStateText(state: ActiveAgent['state']): string {
  switch (state) {
    case 'idle':
      return 'Standing by...'
    case 'thinking':
      return 'Pondering...'
    case 'working':
      return 'Coding away...'
    case 'testing':
      return 'Checking work...'
    case 'success':
      return 'Nailed it!'
    case 'error':
      return 'Trying plan B...'
    case 'struggling':
      return 'Being persistent...'
    default:
      return 'Busy...'
  }
}

// Get state color
function getStateColor(state: ActiveAgent['state']): string {
  switch (state) {
    case 'success':
      return 'text-neo-done'
    case 'error':
      return 'text-neo-pending'  // Yellow - just pivoting, not a real error
    case 'struggling':
      return 'text-orange-500'   // Orange - working hard, being persistent
    case 'working':
    case 'testing':
      return 'text-neo-progress'
    case 'thinking':
      return 'text-neo-pending'
    default:
      return 'text-neo-text-secondary'
  }
}

// Get agent type badge config
function getAgentTypeBadge(agentType: AgentType): { label: string; className: string; icon: typeof Code } {
  if (agentType === 'testing') {
    return {
      label: 'TEST',
      className: 'bg-purple-100 text-purple-700 border-purple-300',
      icon: FlaskConical,
    }
  }
  // Default to coding
  return {
    label: 'CODE',
    className: 'bg-blue-100 text-blue-700 border-blue-300',
    icon: Code,
  }
}

export function AgentCard({ agent, onShowLogs }: AgentCardProps) {
  const isActive = ['thinking', 'working', 'testing'].includes(agent.state)
  const hasLogs = agent.logs && agent.logs.length > 0
  const typeBadge = getAgentTypeBadge(agent.agentType || 'coding')
  const TypeIcon = typeBadge.icon

  return (
    <div
      className={`
        neo-card p-3 min-w-[180px] max-w-[220px]
        ${isActive ? 'animate-pulse-neo' : ''}
        transition-all duration-300
      `}
    >
      {/* Agent type badge */}
      <div className="flex justify-end mb-1">
        <span
          className={`
            inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-bold
            uppercase tracking-wide rounded border
            ${typeBadge.className}
          `}
        >
          <TypeIcon size={10} />
          {typeBadge.label}
        </span>
      </div>

      {/* Header with avatar and name */}
      <div className="flex items-center gap-2 mb-2">
        <AgentAvatar name={agent.agentName} state={agent.state} size="sm" />
        <div className="flex-1 min-w-0">
          <div className="font-display font-bold text-sm truncate">
            {agent.agentName}
          </div>
          <div className={`text-xs ${getStateColor(agent.state)}`}>
            {getStateText(agent.state)}
          </div>
        </div>
        {/* Log button */}
        {hasLogs && onShowLogs && (
          <button
            onClick={() => onShowLogs(agent.agentIndex)}
            className="p-1 hover:bg-neo-bg-secondary rounded transition-colors"
            title={`View logs (${agent.logs?.length || 0} entries)`}
          >
            <ScrollText size={14} className="text-neo-text-secondary" />
          </button>
        )}
      </div>

      {/* Feature info */}
      <div className="mb-2">
        <div className="text-xs text-neo-text-secondary mb-0.5">
          Feature #{agent.featureId}
        </div>
        <div className="text-sm font-medium truncate" title={agent.featureName}>
          {agent.featureName}
        </div>
      </div>

      {/* Thought bubble */}
      {agent.thought && (
        <div className="relative mt-2 pt-2 border-t-2 border-neo-border/30">
          <div className="flex items-start gap-1.5">
            <MessageCircle size={14} className="text-neo-progress shrink-0 mt-0.5" />
            <p
              className="text-xs text-neo-text-secondary line-clamp-2 italic"
              title={agent.thought}
            >
              {agent.thought}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

// Log viewer modal component
interface AgentLogModalProps {
  agent: ActiveAgent
  logs: AgentLogEntry[]
  onClose: () => void
}

export function AgentLogModal({ agent, logs, onClose }: AgentLogModalProps) {
  const [copied, setCopied] = useState(false)
  const typeBadge = getAgentTypeBadge(agent.agentType || 'coding')
  const TypeIcon = typeBadge.icon

  const handleCopy = async () => {
    const logText = logs
      .map(log => `[${log.timestamp}] ${log.line}`)
      .join('\n')
    await navigator.clipboard.writeText(logText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getLogColor = (type: AgentLogEntry['type']) => {
    switch (type) {
      case 'error':
        return 'text-neo-danger'
      case 'state_change':
        return 'text-neo-progress'
      default:
        return 'text-neo-text'
    }
  }

  // Use portal to render modal at document body level (avoids overflow:hidden issues)
  return createPortal(
    <div
      className="fixed inset-0 flex items-center justify-center p-4 bg-black/50"
      style={{ zIndex: 9999 }}
      onClick={(e) => {
        // Close when clicking backdrop
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className="neo-card w-full max-w-4xl max-h-[80vh] flex flex-col bg-neo-bg">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b-3 border-neo-border">
          <div className="flex items-center gap-3">
            <AgentAvatar name={agent.agentName} state={agent.state} size="sm" />
            <div>
              <div className="flex items-center gap-2">
                <h2 className="font-display font-bold text-lg">
                  {agent.agentName} Logs
                </h2>
                <span
                  className={`
                    inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-bold
                    uppercase tracking-wide rounded border
                    ${typeBadge.className}
                  `}
                >
                  <TypeIcon size={10} />
                  {typeBadge.label}
                </span>
              </div>
              <p className="text-sm text-neo-text-secondary">
                Feature #{agent.featureId}: {agent.featureName}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="neo-button neo-button-sm flex items-center gap-1"
              title="Copy all logs"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
              {copied ? 'Copied!' : 'Copy'}
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-neo-bg-secondary rounded transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Log content */}
        <div className="flex-1 overflow-auto p-4 bg-neo-bg-secondary font-mono text-xs">
          {logs.length === 0 ? (
            <p className="text-neo-text-secondary italic">No logs available</p>
          ) : (
            <div className="space-y-1">
              {logs.map((log, idx) => (
                <div key={idx} className={`${getLogColor(log.type)} whitespace-pre-wrap break-all`}>
                  <span className="text-neo-muted">
                    [{new Date(log.timestamp).toLocaleTimeString()}]
                  </span>{' '}
                  {log.line}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t-2 border-neo-border/30 text-xs text-neo-text-secondary">
          {logs.length} log entries
        </div>
      </div>
    </div>,
    document.body
  )
}
