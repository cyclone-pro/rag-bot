-- ============================================
-- Migration 001: Create Interviews Main Table
-- Optimized for 100+ concurrent interviews
-- ============================================

-- Main interviews table (stores everything in one place)
CREATE TABLE IF NOT EXISTS interviews (
    -- Primary key
    interview_id VARCHAR(64) PRIMARY KEY,
    
    -- Candidate information
    candidate_id VARCHAR(64) NOT NULL,
    candidate_name VARCHAR(255),
    candidate_email VARCHAR(255),
    candidate_phone VARCHAR(20),
    
    -- Job description information
    jd_id VARCHAR(64) NOT NULL,
    job_title VARCHAR(255),
    
    -- Interview lifecycle status
    status VARCHAR(20) NOT NULL DEFAULT 'initiated',
    -- Status values: initiated, calling, ringing, in_progress, completed, failed, cancelled
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    
    -- Call infrastructure details
    livekit_room_name VARCHAR(255),
    livekit_dispatch_id VARCHAR(255),
    telnyx_call_sid VARCHAR(255),
    worker_id VARCHAR(64),
    
    -- Full conversation transcript (JSONB - batch written at end)
    transcript JSONB,
    -- Structure: [
    --   {"speaker": "agent", "text": "...", "timestamp": "...", "question_index": 1},
    --   {"speaker": "candidate", "text": "...", "timestamp": "...", "duration_seconds": 45}
    -- ]
    
    -- Interview metadata
    questions_asked INTEGER DEFAULT 0,
    questions_completed INTEGER DEFAULT 0,
    
    -- Evaluation results (computed after interview)
    evaluation_score DECIMAL(3,2), -- 0.00 to 1.00
    evaluation_summary TEXT,
    technical_skills_discussed TEXT[], -- Array of skills
    
    -- Skills match analysis
    skills_coverage JSONB, -- {"required": ["Python", "FastAPI"], "demonstrated": ["Python"], "missing": ["FastAPI"]}
    
    -- Recording (if enabled)
    recording_url TEXT,
    recording_duration_seconds INTEGER,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Milvus sync status
    milvus_synced BOOLEAN DEFAULT FALSE,
    milvus_sync_at TIMESTAMP,
    
    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- Indexes for Performance (with safety checks)
-- ============================================

-- Drop existing indexes first (safe - won't error if they don't exist)
DROP INDEX IF EXISTS idx_interviews_status;
DROP INDEX IF EXISTS idx_interviews_candidate;
DROP INDEX IF EXISTS idx_interviews_created;
DROP INDEX IF EXISTS idx_interviews_completed;
DROP INDEX IF EXISTS idx_interviews_worker;
DROP INDEX IF EXISTS idx_interviews_failed;
DROP INDEX IF EXISTS idx_interviews_milvus_pending;

-- Create indexes
-- Status queries (most common - active interviews)
CREATE INDEX idx_interviews_status ON interviews(status) WHERE status IN ('calling', 'in_progress');

-- Candidate lookups
CREATE INDEX idx_interviews_candidate ON interviews(candidate_id);

-- Date range queries
CREATE INDEX idx_interviews_created ON interviews(created_at DESC);
CREATE INDEX idx_interviews_completed ON interviews(completed_at DESC) WHERE completed_at IS NOT NULL;

-- Worker tracking
CREATE INDEX idx_interviews_worker ON interviews(worker_id) WHERE worker_id IS NOT NULL;

-- Failed interview debugging
CREATE INDEX idx_interviews_failed ON interviews(status, error_message) WHERE status = 'failed';

-- Milvus sync tracking
CREATE INDEX idx_interviews_milvus_pending ON interviews(milvus_synced) WHERE milvus_synced = FALSE;

-- ============================================
-- Trigger for updated_at
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_interviews_updated_at
    BEFORE UPDATE ON interviews
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Modify interview_transcripts (make optional)
-- ============================================

-- Add batch insert flag
ALTER TABLE interview_transcripts 
ADD COLUMN IF NOT EXISTS batch_inserted BOOLEAN DEFAULT FALSE;

-- Add index for batch processing
CREATE INDEX IF NOT EXISTS idx_transcripts_batch 
ON interview_transcripts(interview_id) 
WHERE batch_inserted = FALSE;

-- Make FK constraint deferred (avoid locks during high concurrency)
ALTER TABLE interview_transcripts 
DROP CONSTRAINT IF EXISTS interview_transcripts_interview_id_fkey;

ALTER TABLE interview_transcripts 
ADD CONSTRAINT interview_transcripts_interview_id_fkey 
FOREIGN KEY (interview_id) REFERENCES interviews(interview_id)
DEFERRABLE INITIALLY DEFERRED;

-- ============================================
-- Audit log optimization
-- ============================================

-- Add batch insert capability
ALTER TABLE audit_log 
ADD COLUMN IF NOT EXISTS batch_inserted BOOLEAN DEFAULT FALSE;

-- Index for async batch processing
CREATE INDEX IF NOT EXISTS idx_audit_batch 
ON audit_log(batch_inserted, timestamp_utc) 
WHERE batch_inserted = FALSE;

-- ============================================
-- Connection settings for high concurrency
-- ============================================

-- Show current settings
SHOW max_connections;
SHOW shared_buffers;

-- Recommended settings (run as superuser):
-- ALTER SYSTEM SET max_connections = 200;
-- ALTER SYSTEM SET shared_buffers = '4GB';
-- ALTER SYSTEM SET effective_cache_size = '12GB';
-- ALTER SYSTEM SET work_mem = '50MB';
-- ALTER SYSTEM SET maintenance_work_mem = '1GB';
-- SELECT pg_reload_conf();

-- ============================================
-- Grant permissions
-- ============================================

GRANT SELECT, INSERT, UPDATE, DELETE ON interviews TO recruiterbrain_user;
GRANT SELECT, INSERT, UPDATE ON interview_transcripts TO recruiterbrain_user;
GRANT SELECT, INSERT ON audit_log TO recruiterbrain_user;
