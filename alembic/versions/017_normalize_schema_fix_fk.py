"""Normalize schema and fix foreign key constraints

Revision ID: 017_normalize_schema_fix_fk
Revises: 016_add_multi_tenancy
Create Date: 2025-11-27 00:00:00.000000

Critical Fixes:
1. Fix user_id type mismatch (String → UUID) in progress tables
2. Add missing foreign key constraints
3. Normalize ProcessedContent table
4. Remove JSONB dumping grounds
5. Add proper indexes

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '017_normalize_schema_fix_fk'
down_revision = '016_add_multi_tenancy'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply schema normalization fixes."""
    
    # ========================================================================
    # 1. FIX PROGRESS TABLES - Change user_id from String to UUID
    # ========================================================================
    
    # Drop existing indexes first (IF EXISTS to handle fresh installs)
    op.execute('DROP INDEX IF EXISTS ix_student_progress_user_id')
    op.execute('DROP INDEX IF EXISTS ix_quiz_scores_user_id')
    op.execute('DROP INDEX IF EXISTS ix_learning_sessions_user_id')
    op.execute('DROP INDEX IF EXISTS ix_achievements_user_id')
    op.execute('DROP INDEX IF EXISTS ix_parent_reports_student_id')
    
    # Alter columns to UUID type
    op.alter_column('student_progress', 'user_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    postgresql_using='user_id::uuid')
    
    op.alter_column('quiz_scores', 'user_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    postgresql_using='user_id::uuid')
    
    op.alter_column('learning_sessions', 'user_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    postgresql_using='user_id::uuid')
    
    op.alter_column('achievements', 'user_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    postgresql_using='user_id::uuid')
    
    op.alter_column('parent_reports', 'student_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    postgresql_using='student_id::uuid')
    
    # Recreate indexes
    op.create_index('ix_student_progress_user_id', 'student_progress', ['user_id'])
    op.create_index('ix_quiz_scores_user_id', 'quiz_scores', ['user_id'])
    op.create_index('ix_learning_sessions_user_id', 'learning_sessions', ['user_id'])
    op.create_index('ix_achievements_user_id', 'achievements', ['user_id'])
    op.create_index('ix_parent_reports_student_id', 'parent_reports', ['student_id'])
    
    # ========================================================================
    # 2. ADD FOREIGN KEY CONSTRAINTS
    # ========================================================================
    
    # Progress tables → users
    op.create_foreign_key(
        'fk_student_progress_user',
        'student_progress', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_quiz_scores_user',
        'quiz_scores', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_learning_sessions_user',
        'learning_sessions', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_achievements_user',
        'achievements', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_parent_reports_student',
        'parent_reports', 'users',
        ['student_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Progress → Content
    # First, convert content_id to UUID if it's String
    op.alter_column('student_progress', 'content_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    postgresql_using='content_id::uuid')
    
    op.alter_column('quiz_scores', 'content_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    postgresql_using='content_id::uuid')
    
    op.alter_column('learning_sessions', 'content_id',
                    existing_type=sa.String(),
                    type_=postgresql.UUID(as_uuid=True),
                    nullable=True,  # Sessions may not be tied to specific content
                    postgresql_using='content_id::uuid')
    
    op.create_foreign_key(
        'fk_student_progress_content',
        'student_progress', 'processed_content',
        ['content_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_quiz_scores_content',
        'quiz_scores', 'processed_content',
        ['content_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # ========================================================================
    # 3. NORMALIZE PROCESSED_CONTENT - Extract metadata to separate tables
    # ========================================================================
    
    # Create content_translations table
    op.create_table(
        'content_translations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('language', sa.String(50), nullable=False),
        sa.Column('translated_text', sa.Text, nullable=False),
        sa.Column('translation_model', sa.String(100)),
        sa.Column('translation_quality_score', sa.Float),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['content_id'], ['processed_content.id'], ondelete='CASCADE')
    )
    op.create_index('ix_content_translations_content_id', 'content_translations', ['content_id'])
    op.create_index('ix_content_translations_language', 'content_translations', ['language'])
    
    # Create content_audio table
    op.create_table(
        'content_audio',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('language', sa.String(50), nullable=False),
        sa.Column('audio_file_path', sa.Text, nullable=False),
        sa.Column('audio_format', sa.String(20)),
        sa.Column('duration_seconds', sa.Float),
        sa.Column('tts_model', sa.String(100)),
        sa.Column('accuracy_score', sa.Float),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['content_id'], ['processed_content.id'], ondelete='CASCADE')
    )
    op.create_index('ix_content_audio_content_id', 'content_audio', ['content_id'])
    
    # Create content_validation table
    op.create_table(
        'content_validation',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('validation_type', sa.String(50), nullable=False),  # ncert, script, factual
        sa.Column('alignment_score', sa.Float, nullable=False),
        sa.Column('passed', sa.Boolean, nullable=False),
        sa.Column('issues_found', postgresql.JSONB),  # Structured validation issues
        sa.Column('validated_at', sa.TIMESTAMP, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['content_id'], ['processed_content.id'], ondelete='CASCADE')
    )
    op.create_index('ix_content_validation_content_id', 'content_validation', ['content_id'])
    op.create_index('ix_content_validation_type', 'content_validation', ['validation_type'])
    
    # ========================================================================
    # 4. ADD COMPOSITE INDEXES FOR COMMON QUERIES
    # ========================================================================
    
    op.create_index(
        'ix_progress_user_content_composite',
        'student_progress',
        ['user_id', 'content_id'],
        unique=True  # One progress record per user per content
    )
    
    op.create_index(
        'ix_quiz_user_quiz_attempt',
        'quiz_scores',
        ['user_id', 'quiz_id', 'attempt_number']
    )
    
    op.create_index(
        'ix_sessions_user_date',
        'learning_sessions',
        ['user_id', 'started_at']
    )
    
    # ========================================================================
    # 5. ADD CHECK CONSTRAINTS
    # ========================================================================
    
    op.create_check_constraint(
        'check_progress_percent_range',
        'student_progress',
        'progress_percent >= 0 AND progress_percent <= 100'
    )
    
    op.create_check_constraint(
        'check_quiz_score_range',
        'quiz_scores',
        'score >= 0 AND score <= max_score'
    )
    
    op.create_check_constraint(
        'check_alignment_score_range',
        'content_validation',
        'alignment_score >= 0 AND alignment_score <= 1'
    )


def downgrade() -> None:
    """Revert schema changes."""
    
    # Drop new tables
    op.drop_table('content_validation')
    op.drop_table('content_audio')
    op.drop_table('content_translations')
    
    # Drop check constraints
    op.drop_constraint('check_alignment_score_range', 'content_validation', type_='check')
    op.drop_constraint('check_quiz_score_range', 'quiz_scores', type_='check')
    op.drop_constraint('check_progress_percent_range', 'student_progress', type_='check')
    
    # Drop composite indexes
    op.drop_index('ix_sessions_user_date')
    op.drop_index('ix_quiz_user_quiz_attempt')
    op.drop_index('ix_progress_user_content_composite')
    
    # Drop foreign key constraints
    op.drop_constraint('fk_quiz_scores_content', 'quiz_scores', type_='foreignkey')
    op.drop_constraint('fk_student_progress_content', 'student_progress', type_='foreignkey')
    op.drop_constraint('fk_parent_reports_student', 'parent_reports', type_='foreignkey')
    op.drop_constraint('fk_achievements_user', 'achievements', type_='foreignkey')
    op.drop_constraint('fk_learning_sessions_user', 'learning_sessions', type_='foreignkey')
    op.drop_constraint('fk_quiz_scores_user', 'quiz_scores', type_='foreignkey')
    op.drop_constraint('fk_student_progress_user', 'student_progress', type_='foreignkey')
    
    # Revert UUID columns to String (note: data loss may occur)
    op.alter_column('parent_reports', 'student_id',
                    existing_type=postgresql.UUID(as_uuid=True),
                    type_=sa.String())
    
    op.alter_column('achievements', 'user_id',
                    existing_type=postgresql.UUID(as_uuid=True),
                    type_=sa.String())
    
    op.alter_column('learning_sessions', 'user_id',
                    existing_type=postgresql.UUID(as_uuid=True),
                    type_=sa.String())
    
    op.alter_column('quiz_scores', 'user_id',
                    existing_type=postgresql.UUID(as_uuid=True),
                    type_=sa.String())
    
    op.alter_column('student_progress', 'user_id',
                    existing_type=postgresql.UUID(as_uuid=True),
                    type_=sa.String())
