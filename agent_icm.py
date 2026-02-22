from dotenv import load_dotenv
import os
import logging
import asyncio
import datetime
from livekit import agents, rtc, api
from livekit.agents import AgentSession, Agent, RoomInputOptions, mcp, function_tool, get_job_context, BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip
from livekit.agents.voice import RunContext
from livekit.plugins import noise_cancellation, silero, deepgram, elevenlabs, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from supabase import create_client, Client
from openai import AsyncOpenAI
from livekit.plugins import cartesia

load_dotenv(".env.local")

# Configuracion del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables de entorno MCP
mcp_token = os.getenv("MCP_TOKEN")
mcp_server_url = os.getenv("MCP_SERVER")
mcp_timeout = int(os.getenv("MCP_TIMEOUT", "10"))
mcp_session_timeout = int(os.getenv("MCP_SESSION_TIMEOUT", "30"))

# Variables de entorno para Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "documents")

# Configuracion de embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
K_TOP = int(os.getenv("K_TOP", "3"))

# Variable de entorno para transferencia de llamadas
transfer_to_number = os.getenv("TRANSFER_TO")


def load_system_prompt() -> str:
    """Carga el system prompt desde archivo y reemplaza la fecha actual."""
    fecha_actual = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=-5))
    ).strftime('%A, %d de %B de %Y, %H:%M %p (UTC-5)')

    prompt_path = os.getenv("AGENT_PROMPT_FILE", "prompts/gabriela.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    return prompt.replace("{fecha_actual}", fecha_actual)


class Assistant(Agent):
    def __init__(self) -> None:
        self.participant: rtc.RemoteParticipant | None = None

        # Inicializar clientes para la base de conocimiento
        self._openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._embeddings_model = EMBEDDING_MODEL
        self._embeddings_dimension = EMBEDDING_DIMENSIONS

        # Inicializar cliente de Supabase
        if SUPABASE_URL and SUPABASE_KEY:
            self._supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            self._supabase_client = None

        super().__init__(
            instructions=load_system_prompt(),
        )

    def set_participant(self, participant: rtc.RemoteParticipant):
        """Configurar el participante SIP para poder gestionar la llamada"""
        self.participant = participant
        logger.info(f"[AGENT] Participante configurado: {participant.identity}")

    async def generate_initial_greeting(self, session):
        """Generar saludo inicial proactivo para llamadas entrantes"""
        logger.info("[AGENT] Generando saludo inicial")
        await asyncio.sleep(1.0)
        await session.generate_reply(
            instructions="Saluda al usuario y ofrecele tu ayuda."
        )
        logger.info("[AGENT] Saludo inicial generado")

    @function_tool()
    async def buscar_en_base_de_conocimiento(self, pregunta: str, ctx: RunContext) -> str:
        """
        Usa siempre esta herramienta para obtener informacion de la base de conocimiento y generar una respuesta al usuario.

        Args:
            pregunta (str): Texto en lenguaje natural donde el usuario expresa su consulta.

        Returns:
            str: Referencias con informacion para brindar una respuesta al usuario.
        """
        logger.info(f"[TOOL] buscar_en_base_de_conocimiento: {pregunta}")

        try:
            if not self._supabase_client:
                return "La base de conocimiento no esta configurada. Por favor, contacta con nuestro servicio al cliente para obtener informacion."

            query_embedding = await self._openai_client.embeddings.create(
                input=[pregunta],
                model=self._embeddings_model,
                dimensions=self._embeddings_dimension,
            )

            embeddings = query_embedding.data[0].embedding

            results = self._supabase_client.rpc(
                'match_documents',
                {
                    'query_embedding': embeddings,
                    'match_count': K_TOP,
                    'filter': {}
                }
            ).execute()

            if results.data and len(results.data) > 0:
                salida = ""
                for i, r in enumerate(results.data, start=1):
                    content = r.get("content", "").replace("\r\n", "\n").strip()
                    similarity = r.get("similarity", 0)
                    salida += (
                        f"{i}. **referencia {i} - Inicio**\n"
                        f"id: {r.get('id', 'N/A')}\n"
                        f"similarity: {similarity:.4f}\n"
                        f"content: {content}\n"
                        f"**fin de referencia {i}**\n\n"
                    )
                return salida
            else:
                logger.info("[Supabase] No se encontraron resultados relevantes.")
                return "No encontre informacion especifica sobre tu consulta. Te recomiendo contactar directamente con nuestro servicio al cliente."

        except Exception as e:
            logger.error(f"[TOOL] Error en buscar_en_base_de_conocimiento: {e}")
            return "Lo siento, estoy teniendo problemas para consultar la informacion. Por favor, intenta de nuevo."

    @function_tool()
    async def transfer_call(self, transfer_to: str, reason: str, ctx: RunContext) -> str:
        """Call this tool if the user wants to speak to a human agent.

        Args:
            transfer_to: El destino de la transferencia (ej: "agente humano", "soporte")
            reason: La razon por la que el usuario quiere ser transferido
        """
        logger.info(f"[TOOL] transfer_call - reason: {reason}")

        if not self.participant:
            logger.error("[TOOL] transfer_call - No hay participante configurado")
            return "Error: No hay participante configurado para transferir"

        if not transfer_to_number:
            logger.warning("[TOOL] transfer_call - No hay numero de transferencia configurado")
            return "No puedo transferir la llamada en este momento. Por favor, contacta directamente con nuestro servicio al cliente."

        logger.info(f"[TOOL] transfer_call - transfiriendo a: {transfer_to_number}")

        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to_number}",
                )
            )
            logger.info("[TOOL] transfer_call - transferencia exitosa")
            return "Transferencia completada exitosamente"

        except Exception as e:
            logger.error(f"[TOOL] transfer_call - error: {e}")
            if "no SIP session" in str(e):
                return "La transferencia solo está disponible en llamadas telefónicas reales."
            return f"Error al transferir la llamada: {str(e)}"

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call or when the conversation is complete"""
        logger.info("[TOOL] end_call iniciada")

        await ctx.session.generate_reply(
            instructions="Agradece al usuario por su tiempo y despidete amablemente. Se breve."
        )

        current_speech = ctx.session.current_speech
        if current_speech:
            try:
                await current_speech.wait_for_playout()
            except Exception as e:
                logger.warning(f"[TOOL] end_call - error esperando speech: {e}")

        await asyncio.sleep(0.5)
        await self.hangup()
        logger.info("[TOOL] end_call completada")
        return "llamada terminada"

    async def hangup(self):
        """Colgar la llamada eliminando la room de LiveKit"""
        job_ctx = get_job_context()
        if job_ctx is None:
            logger.warning("[HANGUP] No job context available")
            return
        try:
            logger.info(f"[HANGUP] Eliminando room: {job_ctx.room.name}")
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )
            logger.info("[HANGUP] Room eliminado exitosamente")
        except Exception as e:
            logger.error(f"[HANGUP] Error: {e}")
            try:
                job_ctx.shutdown()
            except Exception:
                pass


async def entrypoint(ctx: agents.JobContext):
    logger.info(f"[ENTRYPOINT] Iniciando - room: {ctx.room.name}")

    # Paso 1: Conectar a la sala de LiveKit
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    logger.info("[ENTRYPOINT] Conectado a la sala")

    # Paso 2: Verificar API keys
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY no esta configurado")
    if not os.getenv("DEEPGRAM_API_KEY"):
        raise ValueError("DEEPGRAM_API_KEY no esta configurado")

    # Paso 3: Configurar servidores MCP
    mcp_servers = []
    if mcp_server_url and mcp_token:
        logger.info(f"[ENTRYPOINT] Configurando MCP: {mcp_server_url}")
        mcp_servers.append(
            mcp.MCPServerHTTP(
                url=mcp_server_url,
                headers={"token": mcp_token},
                timeout=mcp_timeout,
                client_session_timeout_seconds=mcp_session_timeout,
            )
        )
    else:
        logger.info("[ENTRYPOINT] MCP no configurado")

    # Paso 4: Crear instancia del agente
    agent = Assistant()

    # Paso 5: Configurar audio de fondo
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.5),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.7),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    # Paso 6: Configurar y crear la sesion de audio optimizada
    logger.info("[ENTRYPOINT] Configurando sesion del agente")
    session = AgentSession(
        # -- Modelos --
        stt=deepgram.STT(model="nova-2", language="es"),
        llm=openai.LLM(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        # tts=elevenlabs.TTS(
        #     model="eleven_flash_v2_5",
        #     voice_id="b2htR0pMe28pYwCY9gnP",
        #     language="es",
        #     streaming_latency=0,
        # ),
        tts=cartesia.TTS(
            model="sonic-3",
            voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c",
            language="es",
        ),

        # -- VAD optimizado --
        vad=silero.VAD.load(
            min_silence_duration=0.15,
            min_speech_duration=0.05,
            activation_threshold=0.2,
            prefix_padding_duration=0.05,
            force_cpu=True,
        ),

        # -- Deteccion de turnos --
        turn_detection=MultilingualModel(),

        # -- Generacion preemptiva --
        preemptive_generation=True,

        # -- Endpointing --
        min_endpointing_delay=0.05,
        max_endpointing_delay=1.5,

        # -- Interrupciones --
        allow_interruptions=True,
        discard_audio_if_uninterruptible=True,
        min_interruption_duration=0.03,
        min_interruption_words=1,
        min_consecutive_speech_delay=0.03,

        # -- Herramientas --
        max_tool_steps=3,
        mcp_servers=mcp_servers,
    )

    # Paso 7: Iniciar la sesion en la room
    logger.info("[ENTRYPOINT] Iniciando sesion")
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    logger.info("[ENTRYPOINT] Sesion iniciada")

    # Paso 8: Iniciar audio de fondo
    await background_audio.start(room=ctx.room, agent_session=session)
    logger.info("[ENTRYPOINT] Audio de fondo iniciado")

    # Paso 9: Esperar participante SIP
    logger.info("[ENTRYPOINT] Esperando participante SIP...")
    participant = await ctx.wait_for_participant()
    agent.set_participant(participant)
    logger.info(f"[ENTRYPOINT] Participante unido: {participant.identity}")

    # Paso 10: Generar saludo inicial proactivo
    await agent.generate_initial_greeting(session)
    logger.info("[ENTRYPOINT] Agente listo y funcionando")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        initialize_process_timeout=120,
        agent_name="autofuturo-ia",
    ))