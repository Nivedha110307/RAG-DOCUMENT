import { Message, SourceChunk } from '../../store/chatStore';
import { SourceCard } from './SourceCard';

interface Props {
  message: Message;
}

function TypingCursor() {
  return (
    <span className="inline-flex gap-0.5 ml-1 align-middle">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1 h-1 rounded-full bg-blue-500 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </span>
  );
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] ${isUser ? 'order-2' : 'order-1'}`}>
        {/* Avatar */}
        <div className={`flex items-center gap-2 mb-1 ${isUser ? 'justify-end' : ''}`}>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {isUser ? 'You' : 'AI Assistant'}
          </span>
          {message.latency_ms && !isUser && (
            <span className="text-xs text-gray-400 dark:text-gray-600">
              {(message.latency_ms / 1000).toFixed(1)}s
            </span>
          )}
        </div>

        {/* Bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-sm'
              : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-sm shadow-sm'
          }`}
        >
          {message.content || (message.isStreaming ? '' : 'No response')}
          {message.isStreaming && <TypingCursor />}
        </div>

        {/* Source citations */}
        {message.sources && message.sources.length > 0 && !message.isStreaming && (
          <div className="mt-2 space-y-1">
            <p className="text-xs text-gray-500 dark:text-gray-400 px-1">
              Sources ({message.sources.length})
            </p>
            {message.sources.map((source, i) => (
              <SourceCard key={source.chunk_id} source={source} index={i + 1} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
