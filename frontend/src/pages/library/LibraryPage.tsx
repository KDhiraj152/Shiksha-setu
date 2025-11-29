import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  Filter, 
  Grid3X3, 
  List, 
  SlidersHorizontal
} from 'lucide-react';
import { PageHeader } from '../../components/patterns/PageHeader';
import { SearchInput } from '../../components/patterns/SearchInput';
import { ContentCard } from '../../components/patterns/ContentCard';
import { EmptyState } from '../../components/patterns/EmptyState';
import { Button } from '../../components/ui/Button/Button';
import { Badge } from '../../components/ui/Badge/Badge';
import { Spinner } from '../../components/ui/Spinner/Spinner';
import { useLibrary, useSearchContent } from '../../hooks';
import { cn } from '../../lib/cn';
import { GRADES, SUBJECTS, LANGUAGES } from '../../lib/constants';

type ViewMode = 'grid' | 'list';

interface Filters {
  subject?: string;
  grade?: number;
  language?: string;
}

/**
 * Library Page - Browse and manage processed content
 */
export function LibraryPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<Filters>({});
  const [showFilters, setShowFilters] = useState(false);
  const [page, setPage] = useState(0);

  // Search or browse
  const isSearching = searchQuery.length > 2;
  
  const { 
    data: searchResults, 
    isLoading: isSearchLoading 
  } = useSearchContent(
    { q: searchQuery, limit: 20 },
    { enabled: isSearching }
  );

  const {
    data: libraryData,
    isLoading: isLibraryLoading,
  } = useLibrary({ limit: 20, offset: page * 20, ...filters });

  // Get items based on mode
  const items = isSearching
    ? searchResults?.results || []
    : libraryData?.items || [];

  const isLoading = isSearching ? isSearchLoading : isLibraryLoading;
  const hasMore = libraryData ? libraryData.has_more : false;

  const activeFiltersCount = Object.values(filters).filter(Boolean).length;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Page Header */}
      <PageHeader
        title="Content Library"
        description="Browse and manage your processed educational content"
        actions={
          <Link to="/app/playground" className="btn btn-gradient">
            Create New
          </Link>
        }
      />

      {/* Search and Filters Bar */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <SearchInput
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onClear={() => setSearchQuery('')}
            placeholder="Search by title, subject, or content..."
            isLoading={isSearching && isSearchLoading}
          />
        </div>

        <div className="flex items-center gap-2">
          {/* Filter Toggle */}
          <Button
            variant={showFilters ? 'secondary' : 'outline'}
            onClick={() => setShowFilters(!showFilters)}
            className="relative"
          >
            <SlidersHorizontal className="w-4 h-4 mr-2" />
            Filters
            {activeFiltersCount > 0 && (
              <Badge 
                variant="primary" 
                size="sm" 
                className="absolute -top-2 -right-2 min-w-[20px] h-5"
              >
                {activeFiltersCount}
              </Badge>
            )}
          </Button>

          {/* View Mode Toggle */}
          <div className="flex border border-border rounded-lg p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={cn(
                'p-2 rounded transition-colors',
                viewMode === 'grid' 
                  ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600' 
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Grid3X3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={cn(
                'p-2 rounded transition-colors',
                viewMode === 'list' 
                  ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600' 
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="card p-4"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Subject Filter */}
            <div>
              <label className="text-sm font-medium text-foreground mb-2 block">
                Subject
              </label>
              <select
                value={filters.subject || ''}
                onChange={(e) => setFilters({ ...filters, subject: e.target.value || undefined })}
                className="w-full input"
              >
                <option value="">All Subjects</option>
                {SUBJECTS.map((subject) => (
                  <option key={subject.value} value={subject.value}>
                    {subject.icon} {subject.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Grade Filter */}
            <div>
              <label className="text-sm font-medium text-foreground mb-2 block">
                Grade
              </label>
              <select
                value={filters.grade || ''}
                onChange={(e) => setFilters({ ...filters, grade: e.target.value ? parseInt(e.target.value, 10) : undefined })}
                className="w-full input"
              >
                <option value="">All Grades</option>
                {GRADES.map((grade) => (
                  <option key={grade.value} value={grade.value}>
                    {grade.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Language Filter */}
            <div>
              <label className="text-sm font-medium text-foreground mb-2 block">
                Language
              </label>
              <select
                value={filters.language || ''}
                onChange={(e) => setFilters({ ...filters, language: e.target.value || undefined })}
                className="w-full input"
              >
                <option value="">All Languages</option>
                {LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.flag} {lang.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {activeFiltersCount > 0 && (
            <div className="mt-4 pt-4 border-t border-border flex justify-end">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFilters({})}
              >
                Clear all filters
              </Button>
            </div>
          )}
        </motion.div>
      )}

      {/* Content Grid/List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <Spinner size="lg" />
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          icon={<Filter className="w-12 h-12" />}
          title={isSearching ? 'No results found' : 'No content yet'}
          description={
            isSearching
              ? 'Try adjusting your search or filters'
              : 'Upload and process your first document to get started'
          }
          action={
            !isSearching ? {
              label: 'Create Content',
              onClick: () => window.location.href = '/app/playground'
            } : undefined
          }
        />
      ) : (
        <>
          <div
            className={cn(
              viewMode === 'grid'
                ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4'
                : 'space-y-3'
            )}
          >
            {items.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <ContentCard
                  id={item.id}
                  title={`${item.subject} - Grade ${item.grade_level}`}
                  preview={item.simplified_text?.substring(0, 150) || 'No preview available'}
                  gradeLevel={item.grade_level}
                  language={item.language}
                  hasAudio={!!item.audio_available}
                  onClick={() => window.location.href = `/app/content/${item.id}`}
                />
              </motion.div>
            ))}
          </div>

          {/* Load More */}
          {hasMore && !isSearching && (
            <div className="flex justify-center pt-6">
              <Button
                variant="outline"
                onClick={() => setPage(p => p + 1)}
              >
                Load More
              </Button>
            </div>
          )}
        </>
      )}
    </motion.div>
  );
}

export default LibraryPage;
