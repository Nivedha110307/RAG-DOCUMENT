import { useState } from 'react';
import { SourceChunk } from '../../store/chatStore';

interface Props {
  source: SourceChunk;
  index: number;
}

export function SourceCard({ source, index }: Props) {
  const [expanded, setExpanded] = useState(false);
  const score = Math.round(source.similarity_score * 100);

  return (
    <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-2 text-xs">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 text-left"
      >
        <span className="flex-shrink-0 w-5 h-5 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-full flex items-center justify-center text-[10px] font-bold">
          {index}
        </span>
        <span className="flex-1 text-gray-700 dark:text-gray-300 font-medium truncate">
          {source.document_name}
        </span>
        {source.page_number && (
          <span className="text-gray-400">p.{source.page_number}</span>
        )}
        <span
          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
            score >= 80
              ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
              : score >= 60
              ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300'
              : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
          }`}
        >
          {score}%
        </span>
        <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <p className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 leading-relaxed line-clamp-6">
          {source.content}
        </p>
      )}
    </div>
  );
}
