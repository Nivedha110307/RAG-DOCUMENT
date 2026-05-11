import { useState, useRef, KeyboardEvent } from 'react';
import { useChatStore } from '../../store/chatStore';
import { useChat } from '../../hooks/useChat';

export function ChatInput() {
  const [query, setQuery] = useState('');
  const { isLoading } = useChatStore();
  const { sendMessage, stopStreaming } = useChat();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async () => {
    if (!query.trim() || isLoading) return;
    const q = query;
    setQuery('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    await sendMessage(q);
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(e.target.value);
    // Auto-resize
    const ta = e.target;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  };

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-900">
      <div className="flex items-end gap-2 max-w-4xl mx-auto">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents... (Enter to send, Shift+Enter for newline)"
          rows={1}
          className="flex-1 resize-none rounded-xl border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400 dark:placeholder-gray-500"
          disabled={isLoading}
        />
        <button
          onClick={isLoading ? stopStreaming : handleSubmit}
          disabled={!isLoading && !query.trim()}
          className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-colors ${
            isLoading
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 dark:disabled:bg-gray-700 text-white disabled:text-gray-400'
          }`}
        >
          {isLoading ? (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 16 16">
              <rect x="3" y="3" width="10" height="10" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19V6m0 0l-7 7m7-7l7 7" />
            </svg>
          )}
        </button>
      </div>
      <p className="text-center text-xs text-gray-400 dark:text-gray-600 mt-2">
        AI can make mistakes. Verify important information with source documents.
      </p>
    </div>
  );
}
