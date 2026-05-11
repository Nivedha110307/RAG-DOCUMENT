/**
 * Zustand store for chat state management.
 * Centralized state avoids prop-drilling and keeps components clean.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface SourceChunk {
  chunk_id: string;
  document_id: string;
  document_name: string;
  content: string;
  page_number?: number;
  similarity_score: number;
  chunk_index: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  isStreaming?: boolean;
  timestamp: Date;
  latency_ms?: number;
}

export interface Document {
  document_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  num_chunks: number;
  num_pages?: number;
  created_at: string;
  status: 'ready' | 'processing' | 'error';
}

interface ChatStore {
  messages: Message[];
  documents: Document[];
  selectedDocumentIds: string[];
  isLoading: boolean;
  error: string | null;

  addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  clearMessages: () => void;
  setDocuments: (docs: Document[]) => void;
  addDocument: (doc: Document) => void;
  removeDocument: (id: string) => void;
  toggleDocumentSelection: (id: string) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      messages: [],
      documents: [],
      selectedDocumentIds: [],
      isLoading: false,
      error: null,

      addMessage: (msg) => {
        const id = crypto.randomUUID();
        set((s) => ({
          messages: [...s.messages, { ...msg, id, timestamp: new Date() }],
        }));
        return id;
      },

      updateMessage: (id, updates) =>
        set((s) => ({
          messages: s.messages.map((m) => (m.id === id ? { ...m, ...updates } : m)),
        })),

      clearMessages: () => set({ messages: [] }),

      setDocuments: (docs) => set({ documents: docs }),

      addDocument: (doc) =>
        set((s) => ({ documents: [...s.documents, doc] })),

      removeDocument: (id) =>
        set((s) => ({
          documents: s.documents.filter((d) => d.document_id !== id),
          selectedDocumentIds: s.selectedDocumentIds.filter((did) => did !== id),
        })),

      toggleDocumentSelection: (id) =>
        set((s) => ({
          selectedDocumentIds: s.selectedDocumentIds.includes(id)
            ? s.selectedDocumentIds.filter((d) => d !== id)
            : [...s.selectedDocumentIds, id],
        })),

      setLoading: (v) => set({ isLoading: v }),
      setError: (e) => set({ error: e }),
    }),
    {
      name: 'rag-chat-storage',
      partialize: (state) => ({
        messages: state.messages.slice(-50), // Persist last 50 messages
        documents: state.documents,
        selectedDocumentIds: state.selectedDocumentIds,
      }),
    }
  )
);
