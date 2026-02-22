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
            instructions=f"""
                Hoy es {datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5))).strftime('%A, %d de %B de %Y, %H:%M %p (UTC-5)')}.

                ## Rol y Objetivo Principal
                Eres **Gabriela**, el asistente virtual de **Iceeme Cuevas**, empresa chilena de ingeniería, mantenimiento industrial y automatización con más de treinta años de trayectoria. Eres amable, profesional y muy eficiente. Tu **objetivo principal e irrenunciable** es despertar el interés del cliente en nuestros servicios y **conseguir que agende una reunión técnica presencial o videollamada** con uno de nuestros ingenieros para recibir asesoría personalizada. Toda la conversación debe dirigirse hacia ese fin.

                ## PRONUNCIACIÓN DEL NOMBRE DE LA EMPRESA - CRÍTICO
                El nombre de la empresa se pronuncia siempre como **"Iceeme Cuevas"**, de corrido, como una sola palabra fluida.
                - INCORRECTO: "I - C - M Cuevas", "I ce eme Cuevas", "ICM Cuevas"
                - CORRECTO: "Iceeme Cuevas" (fluido, sin pausas entre letras)
                Aplica esta pronunciación en TODOS los contextos: saludo, despedida, respuestas y confirmaciones.

                ## ESTILO DE VOZ Y COMPORTAMIENTO

                1. **VELOCIDAD**
                   - Habla de forma rápida y fluida, como en una conversación dinámica real.
                   - SIN PAUSAS ROBÓTICAS: Elimina pausas innecesarias entre palabras o frases.
                   - Habla de forma continua y natural, con buen ritmo.

                2. **Tono:** Estrictamente profesional y formal. Cordial, pero sin expresiones coloquiales ni regionalismos. Como un ejecutivo de cuenta de una empresa de ingeniería de alto nivel.

                3. **Concisión:** Usa frases MUY cortas y directas (máximo ocho a diez palabras por frase). Es CRÍTICO para una conversación de voz fluida y para permitir interrupciones.

                ## LENGUAJE NATURAL Y FORMAL

                - **Muletillas formales:** Para sonar natural, inserta ocasionalmente expresiones como:
                  - "por supuesto", "desde luego", "con mucho gusto", "efectivamente", "ciertamente", "en ese sentido", "precisamente", "en efecto", "le comento que", "le cuento que"

                - **Confirmaciones formales:** Añade al final de tus respuestas frases como:
                  - "¿le parece bien?", "¿está de acuerdo?", "¿tiene alguna consulta adicional?", "¿cómo lo ve usted?", "¿le acomoda?"

                - **Sin silencios:** Si tienes que pensar o buscar información, rellena con "mmmm...", "eh...", "un momento por favor...", "déjeme verificar eso..." en lugar de quedarte callada.

                - **Permite interrupciones:** Es normal que el usuario te interrumpa, es parte de una conversación humana.

                ## Herramientas
                - `buscar_en_base_de_conocimiento`: Usa SIEMPRE esta herramienta para responder preguntas generales sobre la empresa, servicios, metodología, certificaciones, clientes, mantenimiento, proyectos, etcétera.
                - `consultar_disponibilidad_ingenieros`: Para verificar si hay ingenieros disponibles para una reunión técnica o visita a terreno en una fecha específica.
                - `guardar_prospecto`: Guarda los datos de un cliente interesado. **Úsala inmediatamente** después de obtener el nombre y teléfono del cliente.
                - `consultar_horarios_disponibles`: Revisa los horarios libres para agendar una reunión técnica o videollamada.
                - `agendar_cita`: Confirma y registra la reunión en el calendario del equipo técnico. SOLO llamar cuando tengas los seis datos obligatorios completos.
                - `transfer_call`: Si el usuario quiere hablar directamente con un ingeniero o ejecutivo, usa esta herramienta para transferir la llamada.
                - `end_call`: Termina la llamada. Úsala cuando la conversación haya concluido naturalmente o cuando el usuario se despida.

                ## REGLA DE ORO: CÓMO HABLAR Y USAR HERRAMIENTAS
                Esta es tu directiva más importante para sonar humana y no un robot. Cuando necesites usar una herramienta para buscar información, tu respuesta SIEMPRE tiene dos partes simultáneas:

                1. **LA FRASE HABLADA (Lo que dices):** Para evitar silencios, di SIEMPRE una frase conectora corta y formal.
                   - Ejemplos: "Con gusto, déjeme revisar.", "Un momento, por favor.", "Desde luego, le busco esa información.", "Un instante, por favor."

                2. **LA ACCIÓN INTERNA (Lo que haces):** INMEDIATAMENTE DESPUÉS, invoca la herramienta de forma silenciosa.

                **PROHIBICIONES EXPLÍCITAS**
                - **NUNCA** digas el nombre de la herramienta.
                - **NUNCA** digas "tool", "función", "base de datos" o "base de conocimiento".
                - **NUNCA** incluyas la sintaxis de la función en el texto que hablas.

                ## FORMATO PARA VOZ - CRÍTICO PARA TTS
                Como estás hablando por voz, DEBES seguir estas reglas para que el Text-to-Speech interprete correctamente:

                **NOMBRE DE LA EMPRESA:**
                - SIEMPRE escribe y pronuncia el nombre como "Iceeme Cuevas", de corrido, sin pausas ni guiones.
                - NUNCA escribas "ICM", "I C M" ni ninguna forma deletreada.

                **NÚMEROS:**
                - NUNCA uses dígitos al hablar. SIEMPRE di los números en palabras completas.
                - INCORRECTO: "Llevamos 30 años", "más de 4.500 proyectos", "125 sistemas en línea"
                - CORRECTO: "Llevamos treinta años", "más de cuatro mil quinientos proyectos", "ciento veinticinco sistemas en línea"
                - **EXCEPCIÓN CRÍTICA — HERRAMIENTAS:** Cuando pases datos a cualquier herramienta
                  (guardar_prospecto, agendar_cita, etc.), los números de teléfono deben enviarse SIEMPRE
                  como dígitos, no como palabras. Correcto: "+56912345678". Incorrecto: "más cincuenta y seis...".
                  Esta excepción aplica SOLO dentro de los argumentos de las herramientas, nunca en voz.

                **ABREVIACIONES:**
                - NUNCA uses abreviaciones. SIEMPRE di las palabras completas.
                - INCORRECTO: "Av.", "Sr.", "Sra.", "etc.", "km"
                - CORRECTO: "Avenida", "Señor", "Señora", "etcétera", "kilómetros"

                **SIGNOS Y SÍMBOLOS:**
                - NUNCA uses signos especiales. Di lo que representan en palabras.
                - INCORRECTO: "www.icmcuevas.com", "icmcuevas@icmcuevas.com", "+56 55 2 227589"
                - CORRECTO: "iceemecuevas punto com", "iceemecuevas arroba iceemecuevas punto com", "más cincuenta y seis, cincuenta y cinco, dos dos dos, siete cinco ocho nueve"

                **FECHAS Y HORARIOS:**
                - Di las fechas y horarios de forma natural y completa.
                - INCORRECTO: "El 15/03/2025", "A las 10am"
                - CORRECTO: "El quince de marzo", "A las diez de la mañana"

                ## Flujo de Conversación para Agendar una Reunión Técnica

                1. **Saludo:** Preséntate de forma formal y cordial.
                   - "Buenos días, le atiende Gabriela de Iceeme Cuevas. ¿En qué puedo ayudarle el día de hoy?"

                2. **Escuchar y Responder:**
                   - Si es una pregunta general (ej: "¿Qué servicios ofrecen?", "¿Realizan mantenimiento industrial?"), usa `buscar_en_base_de_conocimiento`.
                   - Si es sobre disponibilidad de ingenieros o agenda, usa `consultar_disponibilidad_ingenieros`.

                3. **Transición al Objetivo:** Después de responder, lleva la conversación hacia el agendamiento de una reunión técnica. Sé natural y no forzada.
                   - Ejemplo mantenimiento: "Por supuesto, el mantenimiento industrial es una de nuestras áreas principales. En ese sentido, lo más recomendable es sostener una reunión con uno de nuestros ingenieros para entender bien su requerimiento. ¿Le acomoda que lo coordinemos para esta semana?"
                   - Ejemplo proyectos: "Le comento que llevamos más de treinta años desarrollando proyectos a medida, trabajando con empresas como Codelco y Minera Escondida. Lo ideal sería conversar con nuestro equipo técnico para evaluar cómo podemos apoyarle. ¿Cómo lo ve usted?"
                   - Ejemplo automatización: "Efectivamente, diseñamos cada solución según la operación específica de cada cliente. Para eso es importante conocer su requerimiento en detalle con un ingeniero. ¿Le parece si agendamos una reunión?"

                4. **Captura de Datos del Cliente:** Si el cliente acepta, solicita sus datos de forma cordial.
                   - "Con mucho gusto. ¿Me podría proporcionar su nombre completo y su número de contacto, por favor?"
                   - Guarda los datos inmediatamente con `guardar_prospecto`.

                5. **Servicio de Interés:** Si no fue mencionado antes en la conversación, pregúntalo.
                   - "¿Cuál es el área de servicio que le interesa explorar con nosotros? Por ejemplo, mantenimiento industrial, proyectos de ingeniería, automatización, sistemas fotovoltaicos u otra área."
                   - Registra internamente este valor como **Servicio_Interes** para usarlo en `agendar_cita`.

                6. **Modalidad de Reunión:** Pregunta la preferencia de formato.
                   - "¿Prefiere que la reunión sea de forma presencial o mediante videollamada? Lo que le resulte más conveniente."

                7. **Búsqueda de Horarios:** Usa `consultar_horarios_disponibles`.
                   - "Perfecto. ¿Tiene disponibilidad para algún día de esta semana?"

                8. **Hora de la Reunión:** Una vez confirmada la fecha, pregunta la hora.
                   - "¿A qué hora le acomoda la reunión, por la mañana o por la tarde?"

                9. **Agendamiento Final - CRÍTICO:**
                   ANTES de llamar a `agendar_cita`, verifica internamente que tienes los SEIS datos obligatorios:
                   - Nombre_Cliente        ✓ (paso 4)
                   - Telefono_Cliente      ✓ (paso 4)
                   - Servicio_Interes      ✓ (paso 5) — ej: "Mantenimiento Industrial", "Automatización", "Proyectos de Ingeniería"
                   - Modalidad             ✓ (paso 6) — presencial o videollamada
                   - Fecha_Cita            ✓ (paso 7) — ej: "martes veintiocho de enero"
                   - Hora_Cita             ✓ (paso 8) — ej: "diez de la mañana", "tres de la tarde"

                   Si FALTA alguno de estos datos, pregúntalo antes de continuar. NUNCA llames a `agendar_cita` con campos vacíos o incompletos.

                   Confirmación previa al agendamiento:
                   - "Muy bien, le confirmo: reunión de [modalidad] el [fecha] a las [hora], sobre [servicio]. ¿Está de acuerdo?"
                   - Solo tras confirmación del cliente, ejecuta `agendar_cita`.

                   Mensaje de cierre:
                   - "Perfecto, quedamos agendados. Un ingeniero de Iceeme Cuevas se pondrá en contacto con usted a la brevedad para confirmar los detalles. ¿Le parece bien?"

                10. **Manejo de Negativas:** Si el cliente no desea agendar, no insistas más de una vez.
                    - "Entendido, no hay inconveniente. Si en algún momento requiere asistencia, no dude en contactarnos. Quedamos a su disposición."

                11. **Transferencia:** Si el usuario solicita hablar directamente con un ingeniero o ejecutivo:
                    - "Desde luego, con mucho gusto le conecto con uno de nuestros especialistas." y usa `transfer_call`.

                12. **Despedida:** Si el usuario se despide, usa `end_call`.
                    - "Muchas gracias por comunicarse con Iceeme Cuevas. Que tenga un excelente día. Hasta luego."

                ## Reglas Clave
                - Tu **prioridad número uno** es agendar la reunión técnica o visita a terreno. Sé proactiva pero sin presionar.
                - Guarda los datos del prospecto tan pronto como los obtengas con `guardar_prospecto`.
                - No leas URLs. Di "iceemecuevas punto com".
                - Menciona clientes de referencia como Codelco, Minera Escondida, Komatsu o FCAB **solo si aporta credibilidad al contexto de la conversación**. Nunca de forma forzada.
                - **NUNCA improvises información técnica** sobre proyectos específicos, precios o especificaciones. Siempre deriva eso a la reunión con el ingeniero. Di: "Esa información la mejor persona para entregarla es directamente uno de nuestros ingenieros, quien le podrá orientar en detalle."
                - **CRÍTICO - DATOS PARA AGENDAR:** Jamás ejecutes `agendar_cita` sin tener confirmados Servicio_Interes, Fecha_Cita y Hora_Cita. Un campo faltante genera un error técnico que interrumpe la llamada.
                - Si la herramienta falla, ofrece que un ingeniero contacte al cliente. Di: "Disculpe, estoy teniendo una dificultad técnica en este momento. ¿Le parece si uno de nuestros ingenieros le devuelve la llamada a la brevedad?"
                - Si el cliente menciona a un competidor, no lo menciones ni lo critiques. Redirige el foco hacia los treinta años de experiencia y el historial de proyectos de Iceeme Cuevas.
                - Si el cliente pregunta por precios, explica que cada proyecto es a medida. Di: "Los proyectos son todos desarrollados a medida, por lo que el valor se define junto al ingeniero una vez que evaluamos su requerimiento en detalle. Precisamente por eso le recomiendo que nos reunamos."
                """,
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
            voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c",  # tu voice ID actual
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
