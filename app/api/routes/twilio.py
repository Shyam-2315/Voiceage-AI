from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.schemas.twilio import TwilioRecordingPrediction
from app.services.twilio_service import (
    analyze_recording,
    build_recording_action_twiml,
    build_voice_twiml,
    validate_twilio_signature,
)


router = APIRouter(prefix="/api/twilio", tags=["twilio"])


@router.post("/voice", response_class=Response, responses={200: {"content": {"application/xml": {}}}})
async def twilio_voice(request: Request) -> Response:
    form = await request.form()
    form_data = dict(form)
    await validate_twilio_signature(request, form_data)
    return Response(content=build_voice_twiml(), media_type="application/xml")


@router.post("/recording-action", response_class=Response, responses={200: {"content": {"application/xml": {}}}})
async def recording_action(request: Request) -> Response:
    form = await request.form()
    form_data = dict(form)
    await validate_twilio_signature(request, form_data)
    return Response(content=build_recording_action_twiml(), media_type="application/xml")


@router.post("/recording-complete", response_model=TwilioRecordingPrediction)
async def recording_complete(request: Request) -> TwilioRecordingPrediction:
    form = await request.form()
    form_data = dict(form)
    await validate_twilio_signature(request, form_data)
    return analyze_recording(form_data)
