#!/usr/bin/env python3
"""Query database to check call processing status."""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Fix DATABASE_URL if it's just an IP
db_url = os.getenv("DATABASE_URL", "")
if db_url and not db_url.startswith("postgresql"):
    # Assume it's just IP, construct full URL
    db_url = f"postgresql://backteam:Airecruiter1_@{db_url}:5432/recruiter_brain"
    os.environ["DATABASE_URL"] = db_url
    print(f"Fixed DATABASE_URL: {db_url[:50]}...")

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    print("Installing psycopg...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg[binary]", "-q"])
    import psycopg
    from psycopg.rows import dict_row


def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set")
        return
    
    print(f"Connecting to: {db_url[:50]}...")
    
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            print("\n" + "=" * 70)
            print("CALL TRANSCRIPTS STATUS")
            print("=" * 70)
            
            # Status breakdown
            cur.execute("""
                SELECT status, COUNT(*) as cnt 
                FROM call_transcripts 
                GROUP BY status 
                ORDER BY cnt DESC
            """)
            for row in cur.fetchall():
                print(f"  {row['status']}: {row['cnt']}")
            
            # Recent calls
            print("\n" + "-" * 70)
            print("RECENT CALLS (last 10)")
            print("-" * 70)
            cur.execute("""
                SELECT call_id, status, error_message, message_count, 
                       created_at, updated_at
                FROM call_transcripts 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            for row in cur.fetchall():
                status_icon = {
                    "parsed": "✅",
                    "failed": "❌",
                    "processing": "⏳",
                    "received": "📥",
                    "skipped": "⏭️",
                    "product_inquiry": "❓"
                }.get(row['status'], "•")
                
                error = f" - {row['error_message'][:50]}..." if row['error_message'] else ""
                print(f"  {status_icon} {row['call_id'][:20]} | {row['status']:15} | msgs:{row['message_count'] or 0:3}{error}")
            
            # Stuck calls (processing for > 10 min)
            print("\n" + "-" * 70)
            print("STUCK CALLS (processing > 10 min)")
            print("-" * 70)
            cur.execute("""
                SELECT call_id, status, created_at, updated_at,
                       EXTRACT(EPOCH FROM (NOW() - created_at))/60 as minutes_ago
                FROM call_transcripts 
                WHERE status = 'processing' 
                  AND created_at < NOW() - INTERVAL '10 minutes'
                ORDER BY created_at
            """)
            stuck = cur.fetchall()
            if stuck:
                for row in stuck:
                    print(f"  ⚠️  {row['call_id']} - stuck for {row['minutes_ago']:.0f} minutes")
            else:
                print("  ✅ No stuck calls")
            
            # Failed calls with errors
            print("\n" + "-" * 70)
            print("RECENT FAILED CALLS")
            print("-" * 70)
            cur.execute("""
                SELECT call_id, error_message, created_at
                FROM call_transcripts 
                WHERE status = 'failed'
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            failed = cur.fetchall()
            if failed:
                for row in failed:
                    print(f"  ❌ {row['call_id'][:20]}")
                    print(f"     Error: {row['error_message'][:100] if row['error_message'] else 'None'}")
            else:
                print("  ✅ No failed calls")
            
            # Check if job_requirements or jd_requirements table exists
            print("\n" + "=" * 70)
            print("JOB REQUIREMENTS")
            print("=" * 70)
            
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_name IN ('job_requirements', 'jd_requirements')
            """)
            tables = [r['table_name'] for r in cur.fetchall()]
            
            if not tables:
                print("  ❌ Neither job_requirements nor jd_requirements table exists!")
                return
            
            job_table = tables[0]
            print(f"  Using table: {job_table}")
            
            # Job counts
            cur.execute(f"SELECT COUNT(*) as cnt FROM {job_table}")
            total = cur.fetchone()['cnt']
            print(f"  Total jobs: {total}")
            
            cur.execute(f"SELECT COUNT(*) as cnt FROM {job_table} WHERE milvus_synced = FALSE")
            unsynced = cur.fetchone()['cnt']
            print(f"  Unsynced to Milvus: {unsynced}")
            
            # Recent jobs
            print("\n" + "-" * 70)
            print("RECENT JOBS (last 10)")
            print("-" * 70)
            cur.execute(f"""
                SELECT job_id, job_title, source_call_id, milvus_synced, created_at
                FROM {job_table}
                ORDER BY created_at DESC
                LIMIT 10
            """)
            for row in cur.fetchall():
                sync_icon = "✅" if row['milvus_synced'] else "❌"
                title = (row['job_title'] or 'No title')[:30]
                call = (row['source_call_id'] or 'N/A')[:15]
                print(f"  {sync_icon} {row['job_id'][:15]} | {title:30} | call: {call}")
            
            # Jobs linked to calls
            print("\n" + "-" * 70)
            print("CALLS WITH JOBS")
            print("-" * 70)
            cur.execute(f"""
                SELECT ct.call_id, ct.status, COUNT(jr.id) as job_count
                FROM call_transcripts ct
                LEFT JOIN {job_table} jr ON jr.source_call_id = ct.call_id
                GROUP BY ct.call_id, ct.status
                HAVING COUNT(jr.id) > 0
                ORDER BY ct.created_at DESC
                LIMIT 10
            """)
            for row in cur.fetchall():
                print(f"  {row['call_id'][:25]} | {row['status']:12} | jobs: {row['job_count']}")
            
            # Processing logs
            print("\n" + "=" * 70)
            print("PROCESSING LOGS (last 24h)")
            print("=" * 70)
            
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'processing_logs'
            """)
            if cur.fetchone():
                cur.execute("""
                    SELECT stage, level, COUNT(*) as cnt
                    FROM processing_logs
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY stage, level
                    ORDER BY 
                        CASE level WHEN 'error' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
                        cnt DESC
                """)
                logs = cur.fetchall()
                if logs:
                    for row in logs:
                        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(row['level'], "•")
                        print(f"  {icon} {row['stage']:30} ({row['level']:7}): {row['cnt']}")
                else:
                    print("  No logs in last 24 hours")
                
                # Recent errors
                print("\n" + "-" * 70)
                print("RECENT ERRORS")
                print("-" * 70)
                cur.execute("""
                    SELECT call_id, stage, message, created_at
                    FROM processing_logs
                    WHERE level = 'error'
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                errors = cur.fetchall()
                if errors:
                    for row in errors:
                        print(f"  ❌ {row['call_id'][:20]} | {row['stage']}")
                        print(f"     {row['message'][:80] if row['message'] else 'No message'}")
                else:
                    print("  ✅ No recent errors")
            else:
                print("  ⚠️ processing_logs table doesn't exist")
            
            # Specific call lookup
            print("\n" + "=" * 70)
            print("SPECIFIC CALL LOOKUP")
            print("=" * 70)
            
            # The call from your screenshot
            test_call_id = "fJFpiorWY1Uoi0Z24P9m"
            cur.execute("""
                SELECT * FROM call_transcripts WHERE call_id = %s
            """, (test_call_id,))
            row = cur.fetchone()
            if row:
                print(f"  Call ID: {row['call_id']}")
                print(f"  Status: {row['status']}")
                print(f"  Error: {row['error_message']}")
                print(f"  Message count: {row['message_count']}")
                print(f"  Created: {row['created_at']}")
                print(f"  Updated: {row['updated_at']}")
                
                # Check for jobs from this call
                cur.execute(f"""
                    SELECT job_id, job_title, milvus_synced 
                    FROM {job_table} 
                    WHERE source_call_id = %s
                """, (test_call_id,))
                jobs = cur.fetchall()
                if jobs:
                    print(f"  Jobs created: {len(jobs)}")
                    for j in jobs:
                        print(f"    - {j['job_id']}: {j['job_title']} (milvus: {j['milvus_synced']})")
                else:
                    print(f"  Jobs created: 0")
            else:
                print(f"  Call {test_call_id} not found")


if __name__ == "__main__":
    main()