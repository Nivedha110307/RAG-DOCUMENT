/**
 * Custom hook for chat interactions.
 * Handles streaming SSE responses with proper cleanup.
 */
import { useCallback, useRef } from 'react';
import { useChatStore, Message } from '../store/chatStore';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export function useChat() {
  const store = useChatStore();
  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (query: string) => {
    if (!query.trim() || store.isLoading) return;

    // Cancel any in-flight request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    store.setError(null);
    store.setLoading(true);

    // Add user message immediately for snappy UX
    store.addMessage({ role: 'user', content: query });

    // Add placeholder assistant message that will be filled by streaming
    const assistantId = store.addMessage({
      role: 'assistant',
      content: '',
      isStreaming: true,
    });

    const startTime = Date.now();

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          query,
          document_ids: store.selectedDocumentIds.length
            ? store.selectedDocumentIds
            : null,
          top_k: 5,
          stream: true,
          chat_history: store.messages
            .slice(-6)
            .filter((m) => !m.isStreaming)
            .map((m) => ({ role: m.role, content: m.content })),
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullContent = '';
      let sources = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));

            if (event.type === 'sources') {
              sources = event.data;
              store.updateMessage(assistantId, { sources });
            } else if (event.type === 'token') {
              fullContent += event.data.replace(/\\n/g, '\n');
              store.updateMessage(assistantId, { content: fullContent });
            } else if (event.type === 'done') {
              const latency = Date.now() - startTime;
              store.updateMessage(assistantId, {
                isStreaming: false,
                latency_ms: latency,
              });
            }
          } catch (_) {
            // Ignore malformed SSE lines
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      const errorMsg = err.message || 'Something went wrong';
      store.setError(errorMsg);
      store.updateMessage(assistantId, {
        content: `Sorry, I encountered an error: ${errorMsg}`,
        isStreaming: false,
      });
    } finally {
      store.setLoading(false);
    }
  }, [store]);

  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
    store.setLoading(false);
  }, [store]);

  return { sendMessage, stopStreaming };
}
