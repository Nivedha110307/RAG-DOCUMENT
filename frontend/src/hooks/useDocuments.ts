/**
 * Hook for document upload and management.
 */
import { useCallback, useState } from 'react';
import { useChatStore } from '../store/chatStore';

const API_BASE = (import.meta as any).env.VITE_API_URL || 'http://localhost:8000/api/v1';

export function useDocuments() {
  const { addDocument, removeDocument, setDocuments } = useChatStore();
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});

  const uploadFile = useCallback(async (file: File): Promise<boolean> => {
    const tempId = crypto.randomUUID();
    setUploadProgress((p) => ({ ...p, [tempId]: 0 }));

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use XMLHttpRequest for upload progress events
      const result = await new Promise<any>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            setUploadProgress((p) => ({
              ...p,
              [tempId]: Math.round((e.loaded / e.total) * 100),
            }));
          }
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(xhr.responseText));
          }
        };
        xhr.onerror = () => reject(new Error('Network error'));
        xhr.open('POST', `${API_BASE}/documents/upload`);
        xhr.send(formData);
      });

      addDocument({
        document_id: result.document_id,
        filename: result.filename,
        file_type: file.name.split('.').pop() || 'unknown',
        file_size_bytes: result.file_size_bytes,
        num_chunks: result.num_chunks,
        num_pages: result.num_pages,
        created_at: result.created_at,
        status: 'ready',
      });

      return true;
    } catch (err: any) {
      console.error('Upload failed:', err);
      return false;
    } finally {
      setUploadProgress((p) => {
        const next = { ...p };
        delete next[tempId];
        return next;
      });
    }
  }, [addDocument]);

  const deleteDocument = useCallback(async (documentId: string): Promise<void> => {
    await fetch(`${API_BASE}/documents/${documentId}`, { method: 'DELETE' });
    removeDocument(documentId);
  }, [removeDocument]);

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents/`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  }, [setDocuments]);

  return { uploadFile, deleteDocument, fetchDocuments, uploadProgress };
}
