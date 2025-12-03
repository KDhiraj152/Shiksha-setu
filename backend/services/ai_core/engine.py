"""
AI Engine - Main Orchestration Layer
=====================================

Central AI engine that coordinates:
- Model loading with M4 hardware optimization
- Context-aware conversation handling
- RAG pipeline integration
- Response formatting and streaming
- Safety checks and validation

NOTE: Safety and filtering behavior is now controlled by the PolicyEngine.
Set ALLOW_UNRESTRICTED_MODE=true to bypass curriculum/educational filters.
See backend/policy/policy_module.py for configuration.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
from functools import lru_cache
import threading

from .context import ConversationContext, ContextManager, ContextRole
from .formatter import ResponseFormatter, FormattedResponse, OutputFormat, SourceReference, Intent
from .router import ModelRouter, ModelTier, TaskType, RoutingDecision

# Import policy module for centralized configuration
try:
    from ...policy import get_policy_engine, PolicyMode
    _POLICY_AVAILABLE = True
except ImportError:
    _POLICY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512  # Reasonable default - most educational answers are <300 tokens
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream: bool = True
    
    # Safety settings - these can be overridden by PolicyEngine
    block_harmful: bool = True
    redact_secrets: bool = True
    
    # Performance settings
    use_cache: bool = True
    timeout_seconds: float = 30.0  # 30s should be plenty for 512 tokens
    
    # RAG settings
    use_rag: Optional[bool] = None  # None = auto-detect, True = force, False = skip


@dataclass
class GenerationResult:
    """Result from text generation."""
    content: str
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float
    model_id: str
    cached: bool = False
    sources: List[SourceReference] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIEngine:
    """
    Production-grade AI Engine for local deployment.
    
    Features:
    - Multi-turn conversation with context management
    - RAG integration for document Q&A
    - M4 hardware-optimized inference
    - Streaming token output
    - Response validation and formatting
    - Caching and graceful degradation
    
    Policy Integration:
    - Uses PolicyEngine for content filtering decisions
    - When ALLOW_UNRESTRICTED_MODE=true, bypasses educational filters
    - System-level safety can still be enabled independently
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        model_router: Optional[ModelRouter] = None,
        formatter: Optional[ResponseFormatter] = None,
        redis_client: Optional[Any] = None,
    ):
        """Initialize the AI Engine."""
        self.context_manager = context_manager or ContextManager(redis_client)
        self.router = model_router or ModelRouter()
        self.formatter = formatter or ResponseFormatter()
        
        # Lazy-loaded components
        self._llm_client = None
        self._rag_service = None
        self._safety_guard = None
        self._policy_engine = None  # PolicyEngine for centralized filtering
        self._profile_service = None
        self._review_queue = None
        self._collaborator = None  # Multi-model validation
        self._sandbox = None        # Code execution sandbox
        
        # Thread safety
        self._lock = threading.Lock()
        self._initialized = False
        
        # Metrics
        self._request_count = 0
        self._total_latency_ms = 0.0
        self._cache_hits = 0
        
        logger.info("AIEngine initialized")
    
    def _ensure_initialized(self):
        """Lazy initialization of components - simplified, no personalization."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            start = time.perf_counter()
            components_loaded = []
            
            # 0. Policy Engine - centralized filtering (load first)
            if _POLICY_AVAILABLE:
                try:
                    self._policy_engine = get_policy_engine()
                    components_loaded.append(f"Policy({self._policy_engine.mode.value})")
                except Exception as e:
                    logger.warning(f"PolicyEngine initialization failed: {e}")
            
            # 1. RAG Service - use singleton, optional don't block on failure
            try:
                from ..rag import get_rag_service
                self._rag_service = get_rag_service()
                components_loaded.append("RAG")
            except Exception as e:
                logger.warning(f"RAG initialization failed (continuing without): {e}")
            
            # 2. Safety Guard - lightweight (now integrates with PolicyEngine)
            try:
                from .safety import SafetyGuard
                self._safety_guard = SafetyGuard()
                components_loaded.append("Safety")
            except Exception as e:
                logger.warning(f"SafetyGuard initialization skipped: {e}")
            
            # NOTE: Personalization removed - no StudentProfile, no grade_level/subject tracking
            # This speeds up initialization significantly
            
            self._initialized = True
            elapsed = (time.perf_counter() - start) * 1000
            
            logger.info(f"AIEngine initialized in {elapsed:.0f}ms: {', '.join(components_loaded) if components_loaded else 'core only'}")
    
    def _get_policy_engine(self):
        """Get the policy engine instance."""
        if self._policy_engine is None and _POLICY_AVAILABLE:
            try:
                self._policy_engine = get_policy_engine()
            except Exception:
                pass
        return self._policy_engine
    
    def _should_block_harmful(self, config: 'GenerationConfig') -> bool:
        """Determine if harmful content blocking should be applied.
        
        Checks PolicyEngine mode first, then falls back to config.
        """
        policy = self._get_policy_engine()
        if policy and policy.mode == PolicyMode.UNRESTRICTED:
            # In unrestricted mode, don't block based on educational/curriculum filters
            # But still respect system-level safety if configured
            return policy.config.block_harmful_content
        return config.block_harmful
    
    def _get_llm_client(self):
        """Get LLM client - uses shared MLX engine."""
        if self._llm_client is not None:
            return self._llm_client
        
        with self._lock:
            if self._llm_client is not None:
                return self._llm_client
            
            # Use shared inference engine (singleton with pre-loaded MLX)
            from ..inference import get_inference_engine
            self._llm_client = get_inference_engine()
            logger.info("Using UnifiedInferenceEngine (MLX) for LLM")
            
            return self._llm_client
    
    def _get_collaborator(self):
        """Get or create Model Collaborator for multi-model validation."""
        if self._collaborator is not None:
            return self._collaborator
        
        with self._lock:
            if self._collaborator is not None:
                return self._collaborator
            
            try:
                from ..pipeline.model_collaboration import get_model_collaborator
                self._collaborator = get_model_collaborator()
                logger.info("AIEngine: Model Collaborator loaded for multi-model validation")
            except Exception as e:
                logger.warning(f"Could not load Model Collaborator: {e}")
            
            return self._collaborator
    
    def _get_sandbox(self):
        """Get or create code execution sandbox."""
        if self._sandbox is not None:
            return self._sandbox
        
        with self._lock:
            if self._sandbox is not None:
                return self._sandbox
            
            try:
                from .sandbox import get_sandbox
                self._sandbox = get_sandbox()
                logger.info("AIEngine: Code sandbox loaded for safe execution")
            except Exception as e:
                logger.warning(f"Could not load sandbox: {e}")
            
            return self._sandbox
    
    def _get_evaluator(self):
        """Get or create Semantic Accuracy Evaluator for validation."""
        if not hasattr(self, '_evaluator'):
            self._evaluator = None
        
        if self._evaluator is not None:
            return self._evaluator
        
        with self._lock:
            if hasattr(self, '_evaluator') and self._evaluator is not None:
                return self._evaluator
            
            try:
                from ..evaluation import get_semantic_evaluator
                self._evaluator = get_semantic_evaluator()
                logger.info("AIEngine: Semantic Evaluator loaded for validation")
            except Exception as e:
                logger.warning(f"Could not load Semantic Evaluator: {e}")
            
            return self._evaluator
    
    def _create_fallback_client(self):
        """Create fallback LLM client for M4."""
        from ..simplify import TransformersClient
        return TransformersClient()
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> FormattedResponse:
        """
        Process a chat message and return formatted response.
        
        Args:
            message: User's message
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID
            config: Generation configuration
            context_data: Additional context (e.g., grade level, language)
        
        Returns:
            FormattedResponse with content, metadata, and sources
        """
        self._ensure_initialized()
        config = config or GenerationConfig()
        start_time = time.perf_counter()
        
        # Get or create conversation context
        if conversation_id:
            context = self.context_manager.get_or_create(
                conversation_id, user_id
            )
        else:
            import uuid
            context = self.context_manager.get_or_create(
                str(uuid.uuid4()), user_id
            )
        
        # Add user message to context
        context.add_message(ContextRole.USER, message)
        
        # NOTE: Personalization removed - no grade_level, subject, learning_style
        # All students get the same quality responses
        
        # Detect intent and route to appropriate handler
        intent = self.formatter.detect_intent(message)
        
        # Build prompt with conversation history
        history = context.get_context_messages(max_messages=10)
        
        # Generate response
        try:
            result = await self._generate_response(
                message=message,
                history=history,
                intent=intent,
                config=config,
                context_data=context_data,  # Pass through context (subject, language, etc.)
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = GenerationResult(
                content=self._get_fallback_response(message, intent),
                tokens_prompt=0,
                tokens_completion=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                model_id="fallback",
                confidence=0.5,
            )
        
        # Add assistant response to context
        context.add_message(ContextRole.ASSISTANT, result.content)
        self.context_manager.save(context)
        
        # NOTE: Review queue removed - no personalization tracking
        
        # Format the response
        formatted = self.formatter.format_response(
            content=result.content,
            query=message,
            sources=result.sources,
            tokens_prompt=result.tokens_prompt,
            tokens_completion=result.tokens_completion,
            latency_ms=result.latency_ms,
            model_id=result.model_id,
            confidence=result.confidence,
        )
        
        # Update metrics
        self._request_count += 1
        self._total_latency_ms += result.latency_ms
        if result.cached:
            self._cache_hits += 1
        
        return formatted
    
    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response token by token.
        
        Yields SSE-formatted events for real-time streaming.
        """
        self._ensure_initialized()
        config = config or GenerationConfig(stream=True)
        start_time = time.perf_counter()
        
        # Get or create conversation context
        import uuid
        conv_id = conversation_id or str(uuid.uuid4())
        context = self.context_manager.get_or_create(conv_id, user_id)
        context.add_message(ContextRole.USER, message)
        
        # Build prompt
        history = context.get_context_messages(max_messages=10)
        intent = self.formatter.detect_intent(message)
        
        # Stream response
        full_response = ""
        
        try:
            async for token in self._stream_generate(
                message=message,
                history=history,
                intent=intent,
                config=config,
                context_data=context_data,
            ):
                full_response += token
                yield self._format_sse_chunk(token)
            
            # Complete event
            latency_ms = (time.perf_counter() - start_time) * 1000
            yield self._format_sse_complete(full_response, latency_ms)
            
            # Save to context
            context.add_message(ContextRole.ASSISTANT, full_response)
            self.context_manager.save(context)
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield self._format_sse_error(str(e))
    
    async def _generate_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        intent: Intent,
        config: GenerationConfig,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate a complete response."""
        start_time = time.perf_counter()
        
        # Route to appropriate model
        task_type = self._intent_to_task_type(intent)
        routing = self.router.route(message, task_type)
        
        # Use dynamic token allocation from routing (unless config explicitly sets lower)
        # This ensures the output is as long as needed for the prompt type
        dynamic_max_tokens = routing.estimated_max_tokens
        if config.max_tokens < dynamic_max_tokens:
            logger.debug(
                f"Upgrading max_tokens from {config.max_tokens} to {dynamic_max_tokens} "
                f"based on prompt analysis"
            )
            config.max_tokens = dynamic_max_tokens
        
        # Check if RAG is needed - respect config.use_rag if explicitly set
        if config.use_rag is not None:
            use_rag = config.use_rag
        else:
            use_rag = self._should_use_rag(message, intent)
        
        sources = []
        augmented_context = ""
        rag_attempted = False
        rag_empty = False
        
        if use_rag and self._rag_service:
            rag_attempted = True
            try:
                rag_result = self._rag_service.search(message, top_k=5, rerank=True)
                if rag_result:
                    sources = [
                        SourceReference(
                            source_id=r.chunk_id,
                            source_type="document",
                            title=r.metadata.get("title", "Document"),
                            location=r.metadata.get("location"),
                            confidence=r.score,
                            quote=r.text[:200] if len(r.text) > 200 else r.text,
                            is_inferred=r.score < 0.8,
                        )
                        for r in rag_result[:3]
                    ]
                    augmented_context = "\n\n".join([r.text for r in rag_result[:3]])
                else:
                    rag_empty = True
                    logger.info(f"RAG returned no results for: {message[:50]}...")
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
                rag_empty = True
        
        # Build prompt with RAG context if available
        system_prompt = self._build_system_prompt(intent, context_data)
        
        # Check policy mode for context modification
        policy = self._get_policy_engine()
        is_unrestricted = policy and policy.mode == PolicyMode.UNRESTRICTED
        
        # Modify system prompt based on RAG results
        if augmented_context:
            # We have verified sources - instruct model to use them
            system_prompt += (
                "\n\nVERIFIED REFERENCE DOCUMENTS PROVIDED:\n"
                "You MUST base your answer primarily on the provided context below. "
                "These are trusted sources. Cite specific information from them.\n"
            )
        elif rag_attempted and rag_empty and not is_unrestricted:
            # RAG was tried but no documents found - make model more cautious (only in restricted mode)
            system_prompt += (
                "\n\nIMPORTANT: No verified reference documents are available for this query.\n"
                "You should:\n"
                "1. Clearly state that you're providing general knowledge, not verified information\n"
                "2. Be extra cautious with specific facts, dates, and numbers\n"
                "3. Recommend the student verify important facts with their textbook or teacher\n"
                "4. Focus on explaining concepts and reasoning rather than specific facts\n"
            )
        
        prompt = self._build_prompt(message, history, augmented_context, system_prompt)
        
        # Generate with LLM
        llm = self._get_llm_client()
        
        try:
            response_text = await self._run_generation(
                llm, prompt, config
            )
            
            # Safety check - uses PolicyEngine to determine if filtering should apply
            if self._should_block_harmful(config) and self._safety_guard:
                response_text = self._safety_guard.filter_response(response_text)
            
            # NOTE: Semantic validation removed - was causing slowness
            # The model quality is sufficient without multi-model verification
            
            # Calculate confidence based on RAG sources
            confidence = 0.7  # Base confidence
            
            if sources:
                # Higher confidence with verified sources
                avg_source_score = sum(s.confidence for s in sources) / len(sources)
                confidence = 0.75 + (avg_source_score * 0.2)  # 0.75-0.95 range
            elif rag_attempted and rag_empty:
                # Lower confidence when RAG found nothing
                confidence = 0.6
            
            # Common uncertainty phrases that reduce confidence
            uncertainty_phrases = [
                "i'm not sure", "i don't know", "i'm not certain",
                "might be", "could be", "possibly", "perhaps",
                "i think", "i believe", "it seems", "general knowledge",
            ]
            # Add educational phrases only in restricted mode
            if not is_unrestricted:
                uncertainty_phrases.extend(["verify with your teacher", "check your textbook"])
            
            response_lower = response_text.lower()
            uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
            if uncertainty_count >= 2:
                confidence = min(confidence, 0.5)  # Model is being appropriately cautious
            elif uncertainty_count == 1:
                confidence = min(confidence, 0.6)
            
            # Factor 4: Response contains specific factual claims without sources
            factual_indicators = [
                "in 1", "in 2", "was born", "died in", "founded in",
                "discovered in", "invented in", "established in",
                "population of", "capital of", "president of", "prime minister of"
            ]
            if not sources and any(ind in response_lower for ind in factual_indicators):
                # Specific factual claims without sources = lower confidence
                confidence = min(confidence, 0.55)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return GenerationResult(
                content=response_text,
                tokens_prompt=len(prompt) // 4,  # Rough estimate
                tokens_completion=len(response_text) // 4,
                latency_ms=latency_ms,
                model_id=routing.model_id,
                sources=sources,
                confidence=confidence,
                metadata={
                    "rag_attempted": rag_attempted,
                    "rag_found_sources": len(sources) > 0,
                    "uncertainty_detected": uncertainty_count > 0,
                }
            )
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def _stream_generate(
        self,
        message: str,
        history: List[Dict[str, str]],
        intent: Intent,
        config: GenerationConfig,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM with RAG support."""
        # Get dynamic token allocation for streaming too
        task_type = self._intent_to_task_type(intent)
        routing = self.router.route(message, task_type)
        
        # Use dynamic token allocation from routing
        dynamic_max_tokens = routing.estimated_max_tokens
        if config.max_tokens < dynamic_max_tokens:
            config.max_tokens = dynamic_max_tokens
        
        # Check if RAG is needed - respect config.use_rag if explicitly set
        if config.use_rag is not None:
            use_rag = config.use_rag
        else:
            use_rag = self._should_use_rag(message, intent)
        
        augmented_context = ""
        rag_empty = False
        
        if use_rag and self._rag_service:
            try:
                rag_result = self._rag_service.search(message, top_k=3, rerank=True)
                if rag_result:
                    augmented_context = "\n\n".join([r.text for r in rag_result[:3]])
                else:
                    rag_empty = True
            except Exception as e:
                logger.warning(f"RAG search failed in streaming: {e}")
                rag_empty = True
        
        # Build system prompt with RAG awareness
        system_prompt = self._build_system_prompt(intent, context_data)
        
        if augmented_context:
            system_prompt += (
                "\n\nVERIFIED REFERENCE DOCUMENTS PROVIDED:\n"
                "Base your answer on the provided context. Cite specific information.\n"
            )
        elif use_rag and rag_empty:
            system_prompt += (
                "\n\nNo verified sources available. Be cautious with specific facts.\n"
            )
        
        prompt = self._build_prompt(message, history, augmented_context, system_prompt)
        
        llm = self._get_llm_client()
        
        # Check if LLM supports streaming
        if hasattr(llm, 'generate_stream'):
            async for token in llm.generate_stream(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ):
                yield token
        else:
            # Fallback: generate full response and simulate streaming
            response = await self._run_generation(llm, prompt, config)
            # Chunk the response for pseudo-streaming
            chunk_size = 4
            for i in range(0, len(response), chunk_size):
                yield response[i:i+chunk_size]
                await asyncio.sleep(0.01)  # Small delay for UX
    
    async def _run_generation(
        self,
        llm: Any,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Run LLM generation with timeout."""
        try:
            # Prefer generate_async for backward compatibility with max_tokens param
            if hasattr(llm, 'generate_async') and asyncio.iscoroutinefunction(llm.generate_async):
                result = await asyncio.wait_for(
                    llm.generate_async(
                        prompt,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    ),
                    timeout=config.timeout_seconds
                )
            elif hasattr(llm, 'generate') and asyncio.iscoroutinefunction(llm.generate):
                # Fallback to generate with config object
                from ..inference import GenerationConfig as InferenceGenConfig
                inf_config = InferenceGenConfig(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                result = await asyncio.wait_for(
                    llm.generate(prompt, inf_config),
                    timeout=config.timeout_seconds
                )
            elif hasattr(llm, 'generate'):
                # Run sync method in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: llm.generate(
                        prompt,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    )
                )
            else:
                # Direct call fallback
                result = str(llm(prompt))
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Generation timed out after {config.timeout_seconds}s")
            return "I apologize, but the response took too long. Please try a simpler question."
    
    def _build_system_prompt(
        self,
        intent: Intent,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build system prompt based on intent and policy mode.
        
        In UNRESTRICTED mode: Acts as a general-purpose AI assistant
        In RESTRICTED mode: Education-focused with verification requirements
        """
        policy = self._get_policy_engine()
        is_unrestricted = policy and policy.mode == PolicyMode.UNRESTRICTED
        
        if is_unrestricted:
            # General-purpose AI assistant prompt (like ChatGPT/Claude)
            base_prompt = (
                "You are a helpful, knowledgeable AI assistant. "
                "You can help with any topic the user asks about.\n\n"
                
                "GUIDELINES:\n"
                "- Be helpful, accurate, and thorough in your responses\n"
                "- Use clear, well-structured explanations\n"
                "- Admit when you're uncertain about something\n"
                "- For code, provide complete, working examples\n"
                "- For math, use LaTeX: inline $x^2$ or block $$\\frac{a}{b}$$\n"
                "- Use markdown for formatting (headers, lists, code blocks)\n\n"
            )
        else:
            # Education-focused prompt with verification requirements
            base_prompt = (
                "You are ShikshaSetu, an intelligent AI assistant. "
                "You MUST provide accurate, factual, and verified information only.\n\n"
                
                "CRITICAL RULES:\n"
                "1. NEVER make up facts, dates, names, or statistics\n"
                "2. If you don't know something with certainty, say 'I'm not certain about this'\n"
                "3. For factual questions, only state what is definitively true\n"
                "4. If context/documents are provided, base your answer on them\n"
                "5. Distinguish between established facts and your reasoning\n"
                "6. For controversial topics, present multiple perspectives\n\n"
                
                "FORMATTING:\n"
                "- Use markdown for structure (headers, lists, bold)\n"
                "- Use LaTeX for math: inline $x^2$ or block $$\\frac{a}{b}$$\n"
                "- Show step-by-step solutions for problems\n"
                "- Use code blocks with language tags for code\n\n"
            )
        
        context_data = context_data or {}
        subject = context_data.get("subject")
        
        # Add subject context if provided
        if subject and subject != "General":
            base_prompt += f"CONTEXT: The user is asking about {subject}.\n\n"
        
        intent_prompts = {
            Intent.CODE_REQUEST: (
                "When providing code:\n"
                "1. Include complete, runnable examples\n"
                "2. Add clear comments explaining the logic\n"
                "3. Suggest tests when appropriate\n"
                "4. Explain the code after the code block\n"
            ),
            Intent.EXPLANATION: (
                "Explain concepts clearly. "
                "Use simple language and examples. "
                "Break down complex topics into digestible parts. "
                "For math topics, include worked examples with LaTeX. "
            ),
            Intent.COMPARISON: (
                "When comparing:\n"
                "1. Create a clear table if comparing multiple items\n"
                "2. List pros and cons for each option\n"
                "3. Provide a recommendation when asked\n"
            ),
            Intent.SMALL_TALK: (
                "Be friendly and conversational while remaining helpful. "
            ),
        }
        
        return base_prompt + intent_prompts.get(intent, "")
    
    def _build_prompt(
        self,
        message: str,
        history: List[Dict[str, str]],
        augmented_context: str,
        system_prompt: str,
    ) -> str:
        """Build the full prompt for the LLM."""
        parts = []
        
        # System prompt
        parts.append(f"<|system|>\n{system_prompt}\n</|system|>")
        
        # RAG context if available
        if augmented_context:
            parts.append(f"\n<|context|>\n{augmented_context}\n</|context|>")
        
        # Conversation history (last few turns)
        for msg in history[-6:]:  # Keep last 3 turns
            role = msg["role"]
            content = msg["content"]
            parts.append(f"\n<|{role}|>\n{content}\n</|{role}|>")
        
        # Current message (if not already in history)
        if not history or history[-1]["content"] != message:
            parts.append(f"\n<|user|>\n{message}\n</|user|>")
        
        # Assistant prompt
        parts.append("\n<|assistant|>\n")
        
        return "".join(parts)
    
    def _intent_to_task_type(self, intent: Intent) -> TaskType:
        """Map intent to task type for routing."""
        mapping = {
            Intent.CODE_REQUEST: TaskType.CODE,
            Intent.EXPLANATION: TaskType.REASONING,
            Intent.COMPARISON: TaskType.REASONING,
            Intent.TRANSLATION: TaskType.TRANSLATION,
            Intent.SIMPLIFICATION: TaskType.SUMMARIZATION,
            Intent.QUESTION: TaskType.CHAT,
            Intent.SMALL_TALK: TaskType.CHAT,
            Intent.TASK: TaskType.REASONING,
        }
        return mapping.get(intent, TaskType.CHAT)
    
    def _should_use_rag(self, message: str, intent: Intent) -> bool:
        """Determine if RAG should be used for this query.
        
        RAG adds latency, so only use when likely to find relevant documents.
        For general knowledge questions, the LLM is sufficient.
        """
        # Only use RAG if the service is available
        if not self._rag_service:
            return False
        
        message_lower = message.lower()
        
        # Skip RAG for simple greetings and meta-questions
        skip_patterns = [
            "hello", "hi ", "hey", "thanks", "thank you", "bye", "goodbye",
            "how are you", "what can you do", "help me", "who are you",
        ]
        if any(pat in message_lower for pat in skip_patterns):
            return False
        
        # Skip RAG for very short queries (likely not educational)
        if len(message) < 15:
            return False
        
        # Keywords that REQUIRE document lookup
        doc_required_keywords = [
            "document", "file", "uploaded", "according to", "from the",
            "based on the", "in my notes", "curriculum", "ncert", "textbook",
            "chapter", "lesson", "syllabus",
        ]
        if any(kw in message_lower for kw in doc_required_keywords):
            return True
        
        # For now, skip RAG for general knowledge to speed up responses
        # The LLM has sufficient knowledge for most educational queries
        # RAG is only needed when user has uploaded specific documents
        return False
    
    def _should_validate(self, message: str, intent: Intent) -> bool:
        """Determine if multi-model validation should be used.
        
        Validation is used for:
        - Factual/educational queries that need accuracy
        - Longer responses that could contain errors
        - Queries without RAG sources (higher hallucination risk)
        """
        # Always validate educational content
        educational_keywords = [
            "explain", "what is", "define", "describe", "how does",
            "calculate", "solve", "formula", "equation", "theorem",
            "history", "science", "math", "physics", "chemistry",
            "biology", "geography", "economics", "concept",
        ]
        message_lower = message.lower()
        if any(kw in message_lower for kw in educational_keywords):
            return True
        
        # Validate for question/explanation intents
        if intent in [Intent.QUESTION, Intent.EXPLANATION, Intent.COMPARISON]:
            return True
        
        return False
    
    def _get_fallback_response(self, _message: str, _intent: Intent) -> str:
        """Generate fallback response when LLM fails.
        
        Keep it simple - no hardcoded responses per intent.
        The LLM should handle everything when available.
        Args:
            _message: User message (reserved for future context-aware fallbacks)
            _intent: Detected intent (reserved for future intent-specific fallbacks)
        """
        return "I'm having trouble processing your request right now. Please try again in a moment."
    
    def _format_sse_chunk(self, text: str) -> str:
        """Format text chunk as SSE event."""
        import json
        return f"data: {json.dumps({'type': 'chunk', 'data': {'text': text}})}\n\n"
    
    def _format_sse_complete(self, text: str, latency_ms: float) -> str:
        """Format completion as SSE event."""
        import json
        import uuid
        return f"data: {json.dumps({'type': 'complete', 'data': {'message_id': str(uuid.uuid4()), 'text': text, 'latency_ms': latency_ms}})}\n\n"
    
    def _format_sse_error(self, error: str) -> str:
        """Format error as SSE event."""
        import json
        return f"data: {json.dumps({'type': 'error', 'data': {'error': error}})}\n\n"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "request_count": self._request_count,
            "avg_latency_ms": self._total_latency_ms / max(1, self._request_count),
            "cache_hit_rate": self._cache_hits / max(1, self._request_count),
            "active_contexts": self.context_manager.get_active_count(),
            "models": self.router.get_stats(),
        }


# Singleton instance
_engine_instance: Optional[AIEngine] = None
_engine_lock = threading.Lock()


def get_ai_engine(redis_client: Optional[Any] = None) -> AIEngine:
    """Get or create the global AI engine instance."""
    global _engine_instance
    
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = AIEngine(redis_client=redis_client)
    
    return _engine_instance
