-- ============================================
-- Add Missing Columns to Existing Interviews Table
-- This adds only what's needed for the new system
-- ============================================

-- Add Milvus sync tracking
ALTER TABLE interviews 
ADD COLUMN IF NOT EXISTS milvus_synced BOOLEAN DEFAULT FALSE;

ALTER TABLE interviews 
ADD COLUMN IF NOT EXISTS milvus_sync_at TIMESTAMP;

-- Add worker tracking (which LiveKit agent handled this)
ALTER TABLE interviews 
ADD COLUMN IF NOT EXISTS worker_id VARCHAR(64);

-- Add LiveKit room reference
ALTER TABLE interviews 
ADD COLUMN IF NOT EXISTS livekit_room_name VARCHAR(255);

-- Add index for Milvus sync status (for batch processing)
DROP INDEX IF EXISTS idx_interviews_milvus_pending;
CREATE INDEX idx_interviews_milvus_pending ON interviews(milvus_synced) 
WHERE milvus_synced = FALSE;

-- Add index for worker tracking
DROP INDEX IF EXISTS idx_interviews_worker;
CREATE INDEX idx_interviews_worker ON interviews(worker_id) 
WHERE worker_id IS NOT NULL;

-- Verify columns were added
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'interviews' 
  AND column_name IN ('milvus_synced', 'milvus_sync_at', 'worker_id', 'livekit_room_name')
ORDER BY column_name;

-- Show total column count
SELECT COUNT(*) as total_columns
FROM information_schema.columns 
WHERE table_name = 'interviews';

-- Success message
DO $$ 
BEGIN 
    RAISE NOTICE '✅ Migration completed!';
    RAISE NOTICE '   Added 4 new columns to existing interviews table';
    RAISE NOTICE '   - milvus_synced';
    RAISE NOTICE '   - milvus_sync_at';
    RAISE NOTICE '   - worker_id';
    RAISE NOTICE '   - livekit_room_name';
END $$;
