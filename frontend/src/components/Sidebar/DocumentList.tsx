import { useChatStore, Document } from '../../store/chatStore';
import { useDocuments } from '../../hooks/useDocuments';

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

function DocIcon({ type }: { type: string }) {
  const colors: Record<string, string> = {
    pdf: 'bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-400',
    docx: 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400',
    txt: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400',
    md: 'bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-400',
  };
  return (
    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${colors[type] || colors.txt}`}>
      {type.toUpperCase()}
    </span>
  );
}

export function DocumentList() {
  const { documents, selectedDocumentIds, toggleDocumentSelection } = useChatStore();
  const { deleteDocument } = useDocuments();

  if (documents.length === 0) {
    return (
      <p className="px-4 py-2 text-xs text-gray-500 dark:text-gray-400">
        No documents uploaded yet
      </p>
    );
  }

  return (
    <div className="space-y-1 px-2">
      {documents.map((doc) => {
        const isSelected = selectedDocumentIds.includes(doc.document_id);
        return (
          <div
            key={doc.document_id}
            className={`flex items-center gap-2 rounded-lg px-2 py-2 transition-colors cursor-pointer group ${
              isSelected
                ? 'bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800'
                : 'hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
            onClick={() => toggleDocumentSelection(doc.document_id)}
          >
            <DocIcon type={doc.file_type} />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-gray-800 dark:text-gray-200 truncate">
                {doc.filename}
              </p>
              <p className="text-[10px] text-gray-400 dark:text-gray-500">
                {doc.num_chunks} chunks · {formatSize(doc.file_size_bytes)}
              </p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); deleteDocument(doc.document_id); }}
              className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-all"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        );
      })}
      {selectedDocumentIds.length > 0 && (
        <p className="text-[10px] text-blue-600 dark:text-blue-400 px-2 pt-1">
          Searching in {selectedDocumentIds.length} selected document{selectedDocumentIds.length !== 1 ? 's' : ''}
        </p>
      )}
    </div>
  );
}
