-- Migration: Add Voice Interview Features
-- Date: 2026-01-03
-- Description: Adds candidate consent tracking and interview analytics view

-- ============================================
-- 1. Create candidate_consents table (GDPR/TCPA compliance)
-- ============================================

CREATE TABLE IF NOT EXISTS candidate_consents (
    consent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id TEXT NOT NULL,
    
    -- What they consented to
    consent_type TEXT NOT NULL, -- 'call_recording', 'ai_interview', 'data_storage'
    consented BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- How consent was obtained
    consent_method TEXT NOT NULL, -- 'web_form', 'verbal', 'email', 'api'
    consent_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Audit trail
    ip_address TEXT,
    user_agent TEXT,
    consent_text TEXT, -- Exact text they agreed to
    consent_version TEXT DEFAULT 'v1.0',
    
    -- Revocation support
    revoked BOOLEAN DEFAULT FALSE,
    revoked_at TIMESTAMPTZ,
    revoked_reason TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(candidate_id, consent_type)
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_consents_candidate ON candidate_consents(candidate_id);
CREATE INDEX IF NOT EXISTS idx_consents_timestamp ON candidate_consents(consent_timestamp);
CREATE INDEX IF NOT EXISTS idx_consents_type ON candidate_consents(consent_type);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_consents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_consents_updated_at ON candidate_consents;
CREATE TRIGGER trigger_consents_updated_at
    BEFORE UPDATE ON candidate_consents
    FOR EACH ROW
    EXECUTE FUNCTION update_consents_updated_at();


-- ============================================
-- 2. Create materialized view for analytics
-- ============================================

-- Drop existing view if it exists
DROP MATERIALIZED VIEW IF EXISTS interview_qa_flat CASCADE;

-- Create flattened view of Q&A pairs
CREATE MATERIALIZED VIEW interview_qa_flat AS
SELECT 
    i.interview_id,
    i.candidate_id,
    i.job_id,
    i.job_title,
    (qa->>'index')::int as question_number,
    qa->>'question' as question,
    qa->>'answer' as answer,
    (qa->>'sentiment')::float as sentiment,
    qa->>'sentiment_label' as sentiment_label,
    qa->'keywords' as keywords,
    qa->>'milvus_id' as milvus_id,
    (qa->>'asked_at')::timestamptz as asked_at,
    (qa->>'answered_at')::timestamptz as answered_at,
    (qa->>'duration_seconds')::float as duration_seconds,
    i.created_at as interview_date,
    i.interview_status,
    i.sentiment_score as overall_sentiment
FROM interviews i
CROSS JOIN LATERAL jsonb_array_elements(i.conversation_log->'qa_pairs') as qa
WHERE i.interview_status = 'completed'
  AND i.conversation_log IS NOT NULL
  AND i.conversation_log ? 'qa_pairs';

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_qa_flat_candidate ON interview_qa_flat(candidate_id);
CREATE INDEX IF NOT EXISTS idx_qa_flat_job ON interview_qa_flat(job_id);
CREATE INDEX IF NOT EXISTS idx_qa_flat_sentiment ON interview_qa_flat(sentiment);
CREATE INDEX IF NOT EXISTS idx_qa_flat_date ON interview_qa_flat(interview_date);
CREATE INDEX IF NOT EXISTS idx_qa_flat_milvus ON interview_qa_flat(milvus_id);
CREATE INDEX IF NOT EXISTS idx_qa_flat_question_num ON interview_qa_flat(question_number);

-- Full-text search index on answers
CREATE INDEX IF NOT EXISTS idx_qa_flat_answer_fts ON interview_qa_flat 
    USING gin(to_tsvector('english', answer));

-- Full-text search index on questions
CREATE INDEX IF NOT EXISTS idx_qa_flat_question_fts ON interview_qa_flat 
    USING gin(to_tsvector('english', question));

-- Refresh function (call after new interviews)
CREATE OR REPLACE FUNCTION refresh_interview_qa_flat()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY interview_qa_flat;
END;
$$ LANGUAGE plpgsql;


-- ============================================
-- 3. Grant permissions (adjust user as needed)
-- ============================================

GRANT SELECT, INSERT, UPDATE, DELETE ON candidate_consents TO backteam;
GRANT SELECT ON interview_qa_flat TO backteam;
GRANT EXECUTE ON FUNCTION refresh_interview_qa_flat() TO backteam;


-- ============================================
-- 4. Verification queries
-- ============================================

-- Check consent table
SELECT 'candidate_consents table created' as status, COUNT(*) as row_count 
FROM candidate_consents;

-- Check materialized view
SELECT 'interview_qa_flat view created' as status, COUNT(*) as row_count 
FROM interview_qa_flat;

-- Done!
SELECT 'Migration completed successfully!' as status;