-- ============================================
-- CLEANUP: Reset Interviews Table
-- Use this if you need to start fresh
-- ============================================

-- WARNING: This will delete ALL interview data!
-- Only run this if you're sure you want to reset.

-- Drop all dependent objects first
DROP TRIGGER IF EXISTS trigger_interviews_updated_at ON interviews CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- Drop all indexes
DROP INDEX IF EXISTS idx_interviews_status CASCADE;
DROP INDEX IF EXISTS idx_interviews_candidate CASCADE;
DROP INDEX IF EXISTS idx_interviews_created CASCADE;
DROP INDEX IF EXISTS idx_interviews_completed CASCADE;
DROP INDEX IF EXISTS idx_interviews_worker CASCADE;
DROP INDEX IF EXISTS idx_interviews_failed CASCADE;
DROP INDEX IF EXISTS idx_interviews_milvus_pending CASCADE;

-- Drop the table
DROP TABLE IF EXISTS interviews CASCADE;

-- Confirmation
DO $$ 
BEGIN 
    RAISE NOTICE '✅ Cleanup completed!';
    RAISE NOTICE '   All interview data has been removed.';
    RAISE NOTICE '   You can now run the migration script again.';
END $$;
