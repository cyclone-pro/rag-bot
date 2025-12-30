-- ============================================
-- SAFE Migration: Create Interviews Table
-- Handles existing tables/indexes gracefully
-- ============================================

-- Step 1: Create table only if it doesn't exist
CREATE TABLE IF NOT EXISTS interviews (
    -- Primary key
    interview_id VARCHAR(64) PRIMARY KEY,
    
    -- Candidate information
    candidate_id VARCHAR(64) NOT NULL,
    candidate_name VARCHAR(255),
    candidate_email VARCHAR(255),
    candidate_phone VARCHAR(20),
    
    -- Job description information
    job_id VARCHAR(64) NOT NULL,
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
    
    -- Interview metadata
    questions_asked INTEGER DEFAULT 0,
    questions_completed INTEGER DEFAULT 0,
    
    -- Evaluation results (computed after interview)
    evaluation_score DECIMAL(3,2),
    evaluation_summary TEXT,
    technical_skills_discussed TEXT[],
    
    -- Skills match analysis
    skills_coverage JSONB,
    
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

-- Step 2: Drop existing indexes (safe - won't error if they don't exist)
DROP INDEX IF EXISTS idx_interviews_status CASCADE;
DROP INDEX IF EXISTS idx_interviews_candidate CASCADE;
DROP INDEX IF EXISTS idx_interviews_created CASCADE;
DROP INDEX IF EXISTS idx_interviews_completed CASCADE;
DROP INDEX IF EXISTS idx_interviews_worker CASCADE;
DROP INDEX IF EXISTS idx_interviews_failed CASCADE;
DROP INDEX IF EXISTS idx_interviews_milvus_pending CASCADE;

-- Step 3: Create fresh indexes
CREATE INDEX idx_interviews_status ON interviews(status) 
WHERE status IN ('calling', 'in_progress');

CREATE INDEX idx_interviews_candidate ON interviews(candidate_id);

CREATE INDEX idx_interviews_created ON interviews(created_at DESC);

CREATE INDEX idx_interviews_completed ON interviews(completed_at DESC) 
WHERE completed_at IS NOT NULL;

CREATE INDEX idx_interviews_worker ON interviews(worker_id) 
WHERE worker_id IS NOT NULL;

CREATE INDEX idx_interviews_failed ON interviews(status, error_message) 
WHERE status = 'failed';

CREATE INDEX idx_interviews_milvus_pending ON interviews(milvus_synced) 
WHERE milvus_synced = FALSE;

-- Step 4: Create/replace trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS trigger_interviews_updated_at ON interviews;

-- Create trigger
CREATE TRIGGER trigger_interviews_updated_at
    BEFORE UPDATE ON interviews
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Step 5: Verify table was created
SELECT 
    'interviews' as table_name,
    COUNT(*) as column_count
FROM information_schema.columns 
WHERE table_name = 'interviews'
GROUP BY table_name;

-- Step 6: Show indexes created
SELECT 
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'interviews'
ORDER BY indexname;

-- Success message
DO $$ 
BEGIN 
    RAISE NOTICE '✅ Migration completed successfully!';
    RAISE NOTICE '   Table: interviews';
    RAISE NOTICE '   Indexes: 7 created';
END $$;
