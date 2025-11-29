/**
 * Reviews Page
 * 
 * Translation review workflow with comments, versions, and approval
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle,
  XCircle,
  MessageSquare,
  Clock,
  History,
  User,
  Search,
  ChevronRight,
  Edit3,
  ArrowLeftRight
} from 'lucide-react';
import { Button } from '../../components/ui/Button/Button';
import { Badge } from '../../components/ui/Badge/Badge';
import { Spinner } from '../../components/ui/Spinner/Spinner';
import { pageVariants, staggerItem, staggerContainer } from '../../lib/animations';
import { reviewService, type Review, type ReviewComment, type ReviewVersion, type ReviewStatus } from '../../services/reviews';

// Status colors
const STATUS_CONFIG: Record<ReviewStatus, { color: string; icon: typeof Clock; label: string }> = {
  pending: { color: 'warning', icon: Clock, label: 'Pending Review' },
  approved: { color: 'success', icon: CheckCircle, label: 'Approved' },
  rejected: { color: 'error', icon: XCircle, label: 'Rejected' },
  revised: { color: 'primary', icon: Edit3, label: 'Needs Revision' },
};

// Review Card
function ReviewCard({ 
  review, 
  onSelect 
}: { 
  review: Review; 
  onSelect: (review: Review) => void;
}) {
  const status = STATUS_CONFIG[review.status] || STATUS_CONFIG.pending;
  const StatusIcon = status.icon;

  return (
    <motion.div
      variants={staggerItem}
      whileHover={{ scale: 1.01 }}
      onClick={() => onSelect(review)}
      className="bg-card rounded-xl border border-border p-6 cursor-pointer hover:shadow-lg transition-shadow"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
            <ArrowLeftRight className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground line-clamp-1">
              {review.content_title || `Translation #${review.id.slice(0, 8)}`}
            </h3>
            <p className="text-sm text-muted-foreground">
              {review.source_lang} → {review.target_lang}
            </p>
          </div>
        </div>
        <Badge variant={status.color as any}>
          <StatusIcon className="w-3 h-3 mr-1" />
          {status.label}
        </Badge>
      </div>

      <div className="flex items-center gap-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-1">
          <User className="w-4 h-4" />
          {review.reviewer_name || 'Unassigned'}
        </div>
        <div className="flex items-center gap-1">
          <MessageSquare className="w-4 h-4" />
          {review.comments_count || 0} comments
        </div>
        <div className="flex items-center gap-1">
          <History className="w-4 h-4" />
          v{review.version || 1}
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-border flex justify-between items-center">
        <span className="text-xs text-muted-foreground">
          Created {new Date(review.created_at).toLocaleDateString()}
        </span>
        <ChevronRight className="w-4 h-4 text-muted-foreground" />
      </div>
    </motion.div>
  );
}

// Review Detail Modal
function ReviewDetail({ 
  review, 
  onClose,
  onUpdate
}: { 
  review: Review; 
  onClose: () => void;
  onUpdate: () => void;
}) {
  const [comments, setComments] = useState<ReviewComment[]>([]);
  const [versions, setVersions] = useState<ReviewVersion[]>([]);
  const [newComment, setNewComment] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [activeTab, setActiveTab] = useState<'comparison' | 'comments' | 'history'>('comparison');

  useEffect(() => {
    loadData();
  }, [review.id]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [commentsData, versionsData] = await Promise.all([
        reviewService.getComments(review.id),
        reviewService.getVersions(review.id),
      ]);
      setComments(commentsData);
      setVersions(versionsData);
    } catch (err) {
      console.error('Failed to load review data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddComment = async () => {
    if (!newComment.trim()) return;

    setSubmitting(true);
    try {
      await reviewService.addComment(review.id, newComment);
      setNewComment('');
      loadData();
    } catch (err) {
      console.error('Failed to add comment:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleApprove = async () => {
    setSubmitting(true);
    try {
      await reviewService.approveReview(review.id);
      onUpdate();
      onClose();
    } catch (err) {
      console.error('Failed to approve:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleReject = async () => {
    const reason = prompt('Please provide a reason for rejection:');
    if (!reason) return;

    setSubmitting(true);
    try {
      await reviewService.rejectReview(review.id, reason);
      onUpdate();
      onClose();
    } catch (err) {
      console.error('Failed to reject:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const status = STATUS_CONFIG[review.status] || STATUS_CONFIG.pending;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.95, y: 20 }}
        onClick={(e) => e.stopPropagation()}
        className="bg-card rounded-2xl border border-border w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="p-6 border-b border-border">
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-xl font-bold text-foreground">
                {review.content_title || `Translation Review`}
              </h2>
              <p className="text-sm text-muted-foreground mt-1">
                {review.source_lang} → {review.target_lang}
              </p>
            </div>
            <Badge variant={status.color as any} className="text-sm">
              {status.label}
            </Badge>
          </div>

          {/* Tabs */}
          <div className="flex gap-4 mt-6">
            {(['comparison', 'comments', 'history'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  activeTab === tab
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
              >
                {tab === 'comparison' && 'Comparison'}
                {tab === 'comments' && `Comments (${comments.length})`}
                {tab === 'history' && `History (${versions.length})`}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Spinner size="lg" />
            </div>
          ) : (
            <>
              {activeTab === 'comparison' && (
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground mb-3">Original</h4>
                    <div className="p-4 bg-muted rounded-lg text-foreground whitespace-pre-wrap">
                      {review.original_text}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground mb-3">Translation</h4>
                    <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg text-foreground whitespace-pre-wrap">
                      {review.translated_text}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'comments' && (
                <div className="space-y-4">
                  {comments.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>No comments yet</p>
                    </div>
                  ) : (
                    comments.map(comment => (
                      <div key={comment.id} className="flex gap-4">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white font-medium">
                          {comment.author_name?.[0] || 'U'}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">
                              {comment.author_name || 'Anonymous'}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {new Date(comment.created_at).toLocaleString()}
                            </span>
                          </div>
                          <p className="text-foreground mt-1">{comment.text}</p>
                        </div>
                      </div>
                    ))
                  )}

                  {/* Add comment */}
                  <div className="mt-6 pt-4 border-t border-border">
                    <textarea
                      value={newComment}
                      onChange={(e) => setNewComment(e.target.value)}
                      placeholder="Add a comment..."
                      className="w-full px-4 py-3 rounded-lg border border-input bg-background text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                      rows={3}
                    />
                    <div className="flex justify-end mt-2">
                      <Button
                        onClick={handleAddComment}
                        disabled={!newComment.trim() || submitting}
                      >
                        {submitting ? <Spinner size="sm" className="mr-2" /> : null}
                        Add Comment
                      </Button>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'history' && (
                <div className="space-y-4">
                  {versions.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>No revision history</p>
                    </div>
                  ) : (
                    versions.map((version, index) => (
                      <div 
                        key={version.id} 
                        className={`p-4 rounded-lg border ${
                          index === 0 ? 'border-primary bg-primary/5' : 'border-border'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-foreground">
                            Version {version.version}
                            {index === 0 && (
                              <Badge variant="primary" className="ml-2">Current</Badge>
                            )}
                          </span>
                          <span className="text-sm text-muted-foreground">
                            {new Date(version.created_at).toLocaleString()}
                          </span>
                        </div>
                        {version.change_summary && (
                          <p className="text-sm text-muted-foreground">
                            {version.change_summary}
                          </p>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-border flex justify-between">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          {review.status === 'pending' && (
            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={handleReject}
                disabled={submitting}
                className="text-error border-error hover:bg-error/10"
              >
                <XCircle className="w-4 h-4 mr-2" />
                Reject
              </Button>
              <Button
                onClick={handleApprove}
                disabled={submitting}
                className="bg-success hover:bg-success/90"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Approve
              </Button>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

export function ReviewsPage() {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedReview, setSelectedReview] = useState<Review | null>(null);
  const [filter, setFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadReviews();
  }, [filter]);

  const loadReviews = async () => {
    setLoading(true);
    try {
      const data = await reviewService.getReviews({
        status: filter === 'all' ? undefined : filter,
      });
      setReviews(data);
    } catch (err) {
      console.error('Failed to load reviews:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredReviews = reviews.filter(review => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      review.content_title?.toLowerCase().includes(query) ||
      review.reviewer_name?.toLowerCase().includes(query) ||
      review.source_lang.toLowerCase().includes(query) ||
      review.target_lang.toLowerCase().includes(query)
    );
  });

  const stats = {
    total: reviews.length,
    pending: reviews.filter(r => r.status === 'pending').length,
    approved: reviews.filter(r => r.status === 'approved').length,
    rejected: reviews.filter(r => r.status === 'rejected').length,
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="container py-8 max-w-6xl mx-auto"
    >
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
            <CheckCircle className="w-6 h-6 text-white" />
          </div>
          Translation Reviews
        </h1>
        <p className="text-muted-foreground mt-2">
          Review and approve translated educational content
        </p>
      </div>

      {/* Stats */}
      <motion.div 
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
      >
        <motion.div variants={staggerItem} className="bg-card rounded-xl border border-border p-4">
          <p className="text-sm text-muted-foreground">Total Reviews</p>
          <p className="text-2xl font-bold text-foreground">{stats.total}</p>
        </motion.div>
        <motion.div variants={staggerItem} className="bg-card rounded-xl border border-border p-4">
          <p className="text-sm text-muted-foreground">Pending</p>
          <p className="text-2xl font-bold text-warning">{stats.pending}</p>
        </motion.div>
        <motion.div variants={staggerItem} className="bg-card rounded-xl border border-border p-4">
          <p className="text-sm text-muted-foreground">Approved</p>
          <p className="text-2xl font-bold text-success">{stats.approved}</p>
        </motion.div>
        <motion.div variants={staggerItem} className="bg-card rounded-xl border border-border p-4">
          <p className="text-sm text-muted-foreground">Rejected</p>
          <p className="text-2xl font-bold text-error">{stats.rejected}</p>
        </motion.div>
      </motion.div>

      {/* Filters */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search reviews..."
            className="w-full pl-10 pr-4 py-3 rounded-lg border border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        <div className="flex gap-2">
          {(['all', 'pending', 'approved', 'rejected'] as const).map(status => (
            <button
              key={status}
              onClick={() => setFilter(status)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                filter === status
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:text-foreground'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Reviews List */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      ) : filteredReviews.length === 0 ? (
        <div className="text-center py-12">
          <CheckCircle className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
          <p className="text-lg text-muted-foreground">
            {searchQuery 
              ? 'No reviews match your search'
              : 'No reviews found'}
          </p>
        </div>
      ) : (
        <motion.div
          variants={staggerContainer}
          initial="initial"
          animate="animate"
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
        >
          {filteredReviews.map(review => (
            <ReviewCard
              key={review.id}
              review={review}
              onSelect={setSelectedReview}
            />
          ))}
        </motion.div>
      )}

      {/* Review Detail Modal */}
      <AnimatePresence>
        {selectedReview && (
          <ReviewDetail
            review={selectedReview}
            onClose={() => setSelectedReview(null)}
            onUpdate={loadReviews}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default ReviewsPage;
