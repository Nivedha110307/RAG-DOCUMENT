import { useState, useCallback, DragEvent } from 'react';
import { useDocuments } from '../../hooks/useDocuments';
import { useChatStore } from '../../store/chatStore';

const ALLOWED_TYPES = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/markdown'];
const ALLOWED_EXTS = ['.pdf', '.txt', '.docx', '.md'];

export function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState<string[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const { uploadFile } = useDocuments();

  const processFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files);
    setErrors([]);

    for (const file of arr) {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!ALLOWED_EXTS.includes(ext)) {
        setErrors((e) => [...e, `${file.name}: unsupported file type`]);
        continue;
      }
      if (file.size > 50 * 1024 * 1024) {
        setErrors((e) => [...e, `${file.name}: file too large (max 50MB)`]);
        continue;
      }

      setUploading((u) => [...u, file.name]);
      const ok = await uploadFile(file);
      setUploading((u) => u.filter((n) => n !== file.name));

      if (!ok) {
        setErrors((e) => [...e, `${file.name}: upload failed`]);
      }
    }
  }, [uploadFile]);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
  };

  return (
    <div className="p-4">
      <div
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        className={`border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${
          isDragging
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
            : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
        }`}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept=".pdf,.txt,.docx,.md"
          className="hidden"
          onChange={(e) => e.target.files && processFiles(e.target.files)}
        />
        <div className="w-7 h-7 mx-auto mb-3 bg-blue-100 dark:bg-blue-900 rounded-xl flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
        </div>
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Drop files here or click to upload
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          PDF, DOCX, TXT, MD · up to 50MB
        </p>
      </div>

      {/* Upload progress */}
      {uploading.map((name) => (
        <div key={name} className="mt-2 flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="truncate">{name}</span>
        </div>
      ))}

      {/* Errors */}
      {errors.map((err) => (
        <p key={err} className="mt-1 text-xs text-red-500">{err}</p>
      ))}
    </div>
  );
}
