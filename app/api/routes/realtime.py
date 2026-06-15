from __future__ import annotations

from fastapi import APIRouter, Request, Response, WebSocket

from app.services.realtime_bridge import build_realtime_voice_twiml, bridge_twilio_stream
from app.services.twilio_service import validate_twilio_signature


router = APIRouter(tags=["realtime"])


@router.post("/api/twilio/realtime-voice", response_class=Response, responses={200: {"content": {"application/xml": {}}}})
async def twilio_realtime_voice(request: Request) -> Response:
    form = await request.form()
    form_data = dict(form)
    await validate_twilio_signature(request, form_data)
    return Response(content=build_realtime_voice_twiml(), media_type="application/xml")


@router.websocket("/api/realtime/twilio-stream")
async def twilio_realtime_stream(websocket: WebSocket) -> None:
    await bridge_twilio_stream(websocket)
