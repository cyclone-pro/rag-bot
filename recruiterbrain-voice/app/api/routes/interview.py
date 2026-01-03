"""
Interview API Routes
Endpoints for managing voice interviews
"""

import logging
import uuid
from datetime import datetime
from typing import Optional
from app.services.telephony import make_interview_call

from fastapi import APIRouter, HTTPException, BackgroundTasks
from livekit import api

from app.config.settings import settings
from app.models.interview import (
    StartInterviewRequest,
    InterviewResponse,
    InterviewStatus,
    InterviewSearchRequest,
    InterviewSearchResponse
)
from app.services.interview_service import start_interview_session
from app.services.database import update_interview_status


logger = logging.getLogger(__name__)

router = APIRouter()


# ==========================================
# Start Interview
# ==========================================

@router.post("/start", response_model=InterviewResponse)
async def start_interview(
    request: StartInterviewRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new voice interview
    
    This endpoint:
    1. Creates interview record in PostgreSQL
    2. Creates LiveKit room
    3. Dispatches voice agent
    4. Initiates phone call via Telnyx
    
    Returns interview details including call SID and room name
    """
    
    try:
        # Generate interview ID
        interview_id = f"interview_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Starting interview: {interview_id}")
        logger.info(f"Candidate: {request.candidate.name}")
        logger.info(f"Position: {request.job_description.title}")
        
        # ==========================================
        # Step 1: Create LiveKit Room
        # ==========================================
        
        #room_name = f"interview-{interview_id}"
        room_name = interview_id 
        
        # Initialize LiveKit API client
        livekit_api = api.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key,
            settings.livekit_api_secret
        )
        
        # Create room
        room = await livekit_api.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=settings.interview_max_duration_seconds + 300,  # Buffer
                max_participants=2,  # Agent + candidate
                metadata=f'{{"interview_id": "{interview_id}"}}'
            )
        )
        
        logger.info(f"Created LiveKit room: {room_name}")
        
        # ==========================================
        # Step 2: Create Interview Session (In-Memory)
        # ==========================================
        
        session = await start_interview_session(
            interview_id=interview_id,
            candidate_data=request.candidate.dict(),
            jd_data=request.job_description.dict(),
            livekit_room=room_name
        )
        
        logger.info(f"Created interview session: {interview_id}")
        
        # ==========================================
        # Step 3: Dispatch Agent to Room
        # ==========================================
        
        # Create agent dispatch
        agent_dispatch = await livekit_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name="interview-agent",
                metadata=f'{{"interview_id": "{interview_id}"}}'
            )
        )
        
        logger.info(f"Dispatched agent to room: {room_name}")
        
  
        # ==========================================
        # Step 4: Create SIP Participant (LiveKit calls directly)
        # ==========================================
        
        # logger.info(f"Creating SIP participant to call {request.candidate.phone_number}")
        
        try:
            # Create SIP participant - LiveKit dials out via Telnyx
            sip_request = api.CreateSIPParticipantRequest(
                sip_trunk_id=settings.livekit_outbound_trunk_id,
                sip_call_to=request.candidate.phone_number,
                room_name=room_name,
                participant_identity=f"candidate-{interview_id}",
                participant_name=request.candidate.name,
                play_dialtone=True,  # Play dial tone while connecting
            )
            
            sip_participant = await livekit_api.sip.create_sip_participant(sip_request)
            
            call_sid = sip_participant.sip_call_id
            call_status = "calling"
            
            logger.info(f"✅ SIP participant created: {call_sid}")
            logger.info(f"   LiveKit is calling {request.candidate.phone_number} via Telnyx")
            
        except Exception as e:
            logger.error(f"Failed to create SIP participant: {e}")
            call_sid = None
            call_status = "failed"
        """
        logger.info(f"Initiating call to {request.candidate.phone_number}")
        
        if settings.telnyx_api_key and settings.telnyx_phone_number:
            call_sid = await make_interview_call(
                phone_number=request.candidate.phone_number,
                livekit_room=room_name,
                interview_id=interview_id
            )
            if call_sid:
                logger.info(f"Call initiated: {call_sid}")
                call_status = "calling"
            else:
                logger.warning("Failed to initiate call")
                call_status = "failed"
                call_sid = None
        else:
            logger.warning("Telnyx not configured - skipping phone call")
            call_sid = None
            call_status = "initiated"
        """

     
        
        # Update status to "calling"
        await update_interview_status(
            interview_id=interview_id,
            interview_status="calling",
            call_status="initiated",
            livekit_room_name=room_name,
            worker_id=agent_dispatch.agent_id if hasattr(agent_dispatch, 'agent_id') else None
        )
        
        # ==========================================
        # Return Response
        # ==========================================
        
        return InterviewResponse(
            interview_id=interview_id,
            status="calling",
            livekit_room_name=room_name,
            call_sid=call_sid,
            created_at=datetime.utcnow(),
            scheduled_time=request.scheduled_time,
            interview_url=f"{settings.livekit_url}/room/{room_name}"
        )
        
    except Exception as e:
        logger.error(f"Error starting interview: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start interview: {str(e)}"
        )


# ==========================================
# Get Interview Status
# ==========================================

@router.get("/status/{interview_id}", response_model=InterviewStatus)
async def get_interview_status(interview_id: str):
    """
    Get current status of an interview
    
    Returns real-time information about:
    - Interview status (initiated, calling, in_progress, completed)
    - Questions asked/completed
    - Duration
    - Results (if completed)
    """
    
    try:
        # Query PostgreSQL for interview
        from app.services.database import get_db_session
        from sqlalchemy import text
        
        async with get_db_session() as session:
            query = text("""
                SELECT 
                    interview_id,
                    interview_status,
                    questions_asked,
                    questions_completed,
                    current_question_index,
                    started_at,
                    completed_at,
                    call_duration_seconds,
                    evaluation_score,
                    recording_url
                FROM interviews
                WHERE interview_id = :interview_id
            """)
            
            result = await session.execute(query, {"interview_id": interview_id})
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Interview not found")
            
            # Calculate estimated completion
            estimated_completion = None
            if row.started_at and row.interview_status == "in_progress":
                from datetime import timedelta
                estimated_completion = row.started_at + timedelta(
                    seconds=settings.interview_max_duration_seconds
                )
            
            return InterviewStatus(
                interview_id=row.interview_id,
                status=row.interview_status,
                questions_asked=row.questions_asked or 0,
                questions_completed=row.questions_completed or 0,
                current_question_index=row.current_question_index or 0,
                started_at=row.started_at,
                estimated_completion=estimated_completion,
                duration_seconds=row.call_duration_seconds,
                evaluation_score=row.evaluation_score,
                recording_url=row.recording_url
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching interview status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch status: {str(e)}"
        )


# ==========================================
# Search Interviews
# ==========================================

@router.post("/search", response_model=InterviewSearchResponse)
async def search_interviews(request: InterviewSearchRequest):
    """
    Semantic search across interview transcripts
    
    Uses Milvus vector search to find similar interviews based on:
    - Question/answer content
    - Skills discussed
    - Technical topics
    """
    
    try:
        from app.services.database import embedding_service, milvus_pool
        from pymilvus import Collection
        import time
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await embedding_service.generate_query_embedding(request.query)
        
        # Get Milvus collection
        collection = milvus_pool.get_collection()
        
        # Build filter expression
        filter_expr = []
        
        if request.candidate_id:
            filter_expr.append(f'candidate_id == "{request.candidate_id}"')
        
        if request.job_id:
            filter_expr.append(f'job_id == "{request.job_id}"')
        
        if request.min_score is not None:
            filter_expr.append(f'evaluation_score >= {request.min_score}')
        
        if request.date_from:
            timestamp = int(request.date_from.timestamp())
            filter_expr.append(f'interview_date >= {timestamp}')
        
        if request.date_to:
            timestamp = int(request.date_to.timestamp())
            filter_expr.append(f'interview_date <= {timestamp}')
        
        expr = " && ".join(filter_expr) if filter_expr else None
        
        # Search
        search_results = collection.search(
            data=[query_embedding],
            anns_field="interview_embedding",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=request.limit,
            offset=request.offset,
            expr=expr,
            output_fields=[
                "interview_id",
                "candidate_id",
                "job_title",
                "evaluation_score",
                "interview_summary",
                "skills_discussed",
                "interview_date"
            ]
        )
        
        # Format results
        results = []
        
        for hits in search_results:
            for hit in hits:
                results.append({
                    "interview_id": hit.entity.get("interview_id"),
                    "candidate_id": hit.entity.get("candidate_id"),
                    "job_title": hit.entity.get("job_title"),
                    "similarity_score": hit.score,
                    "evaluation_score": hit.entity.get("evaluation_score"),
                    "summary": hit.entity.get("interview_summary", ""),
                    "skills_discussed": hit.entity.get("skills_discussed", "").split(", "),
                    "interview_date": datetime.fromtimestamp(hit.entity.get("interview_date", 0)),
                    "interview_url": f"/interview/{hit.entity.get('interview_id')}"
                })
        
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return InterviewSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),  # For pagination, query collection.num_entities
            limit=request.limit,
            offset=request.offset,
            search_time_ms=search_time
        )
    
    except Exception as e:
        logger.error(f"Error searching interviews: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ==========================================
# Cancel Interview
# ==========================================

@router.post("/cancel/{interview_id}")
async def cancel_interview(interview_id: str):
    """
    Cancel an ongoing interview
    """
    
    try:
        # Update status
        await update_interview_status(
            interview_id=interview_id,
            interview_status="cancelled",
            call_status="cancelled"
        )
        
        return {"message": "Interview cancelled successfully"}
    
    except Exception as e:
        logger.error(f"Error cancelling interview: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel: {str(e)}"
        )
