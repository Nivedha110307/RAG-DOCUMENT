import { DropZone } from '../Upload/DropZone';
import { DocumentList } from './DocumentList';

export function Sidebar() {
  return (
    <aside className="w-64 flex-shrink-0 border-r border-gray-200 dark:border-gray-800 flex flex-col bg-white dark:bg-gray-950 h-full">
      {/* Header / Logo Section */}
      <div className="p-4 mb-2">
        <div className="flex items-center gap-2.5">
          <div className="w-5 h-5 bg-blue-600 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm shadow-blue-500/20">
            <svg 
              className="w-3 h-3 text-white" 
              fill="currentColor" 
              viewBox="0 0 20 20"
            >
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
              <path 
                fillRule="evenodd" 
                d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" 
                clipRule="evenodd"
              />
            </svg>
          </div>
          <h1 className="font-bold text-base tracking-tight text-gray-900 dark:text-gray-100">
            DocuMind AI
          </h1>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto px-4 space-y-6">
        <section>
          <DropZone />
        </section>

        <section>
          <h3 className="px-1 text-[11px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-3">
            Your Documents
          </h3>
          <DocumentList />
        </section>
      </div>

      {/* Footer Section */}
      <div className="p-4 mt-auto border-t border-gray-100 dark:border-gray-800">
        <div className="flex flex-col items-center gap-1">
          <p className="text-[10px] font-medium text-gray-400 dark:text-gray-500 uppercase tracking-tight">
            Powered by
          </p>
          <div className="flex items-center gap-1.5 opacity-60 grayscale hover:grayscale-0 transition-all">
             <span className="text-[10px] font-semibold text-gray-600 dark:text-gray-400">GPT-4 + CHROMADB</span>
          </div>
        </div>
      </div>
    </aside>
  );
}