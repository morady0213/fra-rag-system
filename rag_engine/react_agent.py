"""
ReAct Agent for Multi-Hop Legal Reasoning.

Implements the ReAct (Reasoning + Acting) pattern for complex legal questions
that require multiple retrieval steps and logical reasoning.

Example:
    Query: "هل تحتاج شركة تأمين برأس مال 30 مليون جنيه إلى موافقة مسبقة؟"
    
    Thought 1: أحتاج معرفة الحد الأدنى لرأس المال لشركات التأمين
    Action 1: retrieve("الحد الأدنى رأس المال شركات التأمين")
    Observation 1: "الحد الأدنى 60 مليون جنيه"
    
    Thought 2: رأس المال 30 مليون أقل من 60 مليون، أحتاج معرفة الاستثناءات
    Action 2: retrieve("استثناءات رأس المال شركات التأمين")
    Observation 2: "لا توجد استثناءات"
    
    Final Answer: "لا، الشركة لا تستوفي الحد الأدنى..."
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from loguru import logger


class ActionType(Enum):
    """Available actions for the ReAct agent."""
    RETRIEVE = "retrieve"      # Search documents
    CALCULATE = "calculate"    # Simple math
    COMPARE = "compare"        # Compare values
    LOOKUP = "lookup"          # Lookup specific entity
    ANSWER = "answer"          # Provide final answer


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    step_number: int
    thought: str
    action: ActionType
    action_input: str
    observation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action.value,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentResult:
    """Result from the ReAct agent."""
    answer: str
    steps: List[AgentStep]
    sources: List[Dict[str, Any]]
    total_retrievals: int
    reasoning_complete: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "sources": self.sources,
            "total_retrievals": self.total_retrievals,
            "reasoning_complete": self.reasoning_complete,
        }
    
    def get_reasoning_trace(self) -> str:
        """Get formatted reasoning trace for display."""
        trace = "## 🧠 مسار الاستدلال (Reasoning Trace)\n\n"
        
        for step in self.steps:
            trace += f"### الخطوة {step.step_number}\n"
            trace += f"**💭 التفكير:** {step.thought}\n"
            trace += f"**🔧 الإجراء:** {step.action.value}({step.action_input[:50]}...)\n"
            trace += f"**👁️ الملاحظة:** {step.observation[:200]}...\n\n"
        
        return trace


class ReActAgent:
    """
    ReAct Agent for multi-hop legal reasoning.
    
    Uses the Thought-Action-Observation loop to break down complex
    legal questions into manageable retrieval and reasoning steps.
    """
    
    SYSTEM_PROMPT_AR = """أنت وكيل ذكي متخصص في تحليل الأسئلة القانونية والتنظيمية المصرية.

مهمتك: الإجابة على أسئلة المستخدم بدقة من خلال البحث في المستندات التنظيمية.

لكل سؤال، اتبع هذا النمط بدقة:

Thought: [تحليلك للموقف الحالي وما تحتاج معرفته للإجابة]
Action: [اختر من: retrieve, calculate, compare, answer]
Action Input: [استعلام البحث أو القيم للحساب أو الإجابة النهائية]

الإجراءات المتاحة:
- retrieve: البحث عن معلومات في المستندات (استخدم استعلام بحث محدد)
- calculate: حساب رياضي بسيط (مثل: 50 - 30 = 20)
- compare: مقارنة قيمتين (مثل: 30 < 50)
- answer: تقديم الإجابة النهائية (فقط عندما تملك معلومات كافية)

قواعد مهمة:
1. ابدأ دائماً بـ Thought لتحليل السؤال
2. لا تخمن - إذا لم تجد المعلومة، ابحث عنها
3. عند الإجابة النهائية، اذكر المصادر بوضوح
4. استمر في البحث حتى تجمع كل المعلومات اللازمة
5. الحد الأقصى 5 خطوات بحث

مثال:
السؤال: ما هو الحد الأدنى لرأس المال لشركة تمويل استهلاكي؟

Thought: أحتاج البحث عن متطلبات رأس المال لشركات التمويل الاستهلاكي
Action: retrieve
Action Input: الحد الأدنى رأس المال شركة تمويل استهلاكي

[بعد الملاحظة]

Thought: وجدت أن الحد الأدنى هو 50 مليون جنيه، لدي معلومات كافية للإجابة
Action: answer
Action Input: الحد الأدنى لرأس المال لشركة التمويل الاستهلاكي هو 50 مليون جنيه مصري [المصدر: نموذج قيد فرع لشركة تمويل الاستهلاكي]
"""

    SYSTEM_PROMPT_EN = """You are an intelligent agent specialized in analyzing Egyptian legal and regulatory questions.

Your task: Answer user questions accurately by searching regulatory documents.

For each question, follow this pattern precisely:

Thought: [Your analysis of the current situation and what you need to know]
Action: [Choose from: retrieve, calculate, compare, answer]
Action Input: [Search query, values for calculation, or final answer]

Available actions:
- retrieve: Search documents (use specific search query)
- calculate: Simple math (e.g., 50 - 30 = 20)
- compare: Compare two values (e.g., 30 < 50)
- answer: Provide final answer (only when you have sufficient information)

Important rules:
1. Always start with Thought to analyze the question
2. Don't guess - if you don't find the information, search for it
3. When providing final answer, clearly cite sources
4. Continue searching until you gather all necessary information
5. Maximum 5 search steps
"""

    def __init__(
        self,
        retriever,
        llm_client,
        max_iterations: int = 5,
        language: str = "ar",
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            retriever: Retriever instance (HybridRetriever or similar)
            llm_client: LLM client for reasoning (GrokClient)
            max_iterations: Maximum reasoning iterations
            language: Language for prompts ("ar" or "en")
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.language = language
        
        self.system_prompt = self.SYSTEM_PROMPT_AR if language == "ar" else self.SYSTEM_PROMPT_EN
        
        logger.info(f"ReActAgent initialized (max_iterations={max_iterations}, language={language})")
    
    def run(self, query: str, k: int = 3) -> AgentResult:
        """
        Run the ReAct agent on a query.
        
        Args:
            query: User's question
            k: Number of documents to retrieve per search
            
        Returns:
            AgentResult with answer, reasoning steps, and sources
        """
        steps: List[AgentStep] = []
        all_sources: List[Dict[str, Any]] = []
        total_retrievals = 0
        
        # Build initial context
        context = f"السؤال: {query}\n\n" if self.language == "ar" else f"Question: {query}\n\n"
        
        for iteration in range(self.max_iterations):
            # Get next action from LLM
            response = self._get_llm_response(context)
            
            # Parse the response
            thought, action, action_input = self._parse_response(response)
            
            if not action:
                logger.warning(f"Could not parse action from response: {response[:200]}")
                # Try to extract answer anyway
                if "answer" in response.lower() or "الإجابة" in response:
                    return AgentResult(
                        answer=response,
                        steps=steps,
                        sources=all_sources,
                        total_retrievals=total_retrievals,
                        reasoning_complete=True,
                    )
                continue
            
            # Execute action
            if action == ActionType.ANSWER:
                # Final answer
                step = AgentStep(
                    step_number=iteration + 1,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation="[Final Answer]",
                )
                steps.append(step)
                
                return AgentResult(
                    answer=action_input,
                    steps=steps,
                    sources=all_sources,
                    total_retrievals=total_retrievals,
                    reasoning_complete=True,
                )
            
            # Execute other actions
            observation = self._execute_action(action, action_input, k)
            total_retrievals += 1 if action == ActionType.RETRIEVE else 0
            
            # Collect sources from retrieval
            if action == ActionType.RETRIEVE and hasattr(self, '_last_retrieval_sources'):
                all_sources.extend(self._last_retrieval_sources)
            
            # Record step
            step = AgentStep(
                step_number=iteration + 1,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)
            
            # Update context for next iteration
            context += f"""
Thought: {thought}
Action: {action.value}
Action Input: {action_input}
Observation: {observation}

"""
            
            logger.info(f"Agent step {iteration + 1}: {action.value} -> {observation[:100]}...")
        
        # Max iterations reached - synthesize answer from gathered info
        final_answer = self._synthesize_answer(query, steps, all_sources)
        
        return AgentResult(
            answer=final_answer,
            steps=steps,
            sources=all_sources,
            total_retrievals=total_retrievals,
            reasoning_complete=False,
        )
    
    def _get_llm_response(self, context: str) -> str:
        """Get response from LLM."""
        if not self.llm_client:
            return "Action: answer\nAction Input: لا يتوفر عميل LLM للاستدلال"
        
        try:
            # Use the LLM client to generate response
            # We need to call the underlying API directly for agent mode
            import requests
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context},
            ]
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_client.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_client.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000,
                },
                timeout=30,
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return ""
    
    def _parse_response(self, response: str) -> Tuple[str, Optional[ActionType], str]:
        """Parse Thought, Action, and Action Input from response."""
        thought = ""
        action = None
        action_input = ""
        
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract Action
        action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).lower().strip()
            try:
                action = ActionType(action_str)
            except ValueError:
                # Try to map common variations
                action_map = {
                    "search": ActionType.RETRIEVE,
                    "find": ActionType.RETRIEVE,
                    "بحث": ActionType.RETRIEVE,
                    "استرجاع": ActionType.RETRIEVE,
                    "حساب": ActionType.CALCULATE,
                    "مقارنة": ActionType.COMPARE,
                    "إجابة": ActionType.ANSWER,
                    "الإجابة": ActionType.ANSWER,
                }
                action = action_map.get(action_str)
        
        # Extract Action Input
        input_match = re.search(r"Action Input:\s*(.+?)(?=Thought:|Observation:|$)", response, re.DOTALL | re.IGNORECASE)
        if input_match:
            action_input = input_match.group(1).strip()
        
        return thought, action, action_input
    
    def _execute_action(self, action: ActionType, action_input: str, k: int) -> str:
        """Execute an action and return observation."""
        if action == ActionType.RETRIEVE:
            return self._action_retrieve(action_input, k)
        elif action == ActionType.CALCULATE:
            return self._action_calculate(action_input)
        elif action == ActionType.COMPARE:
            return self._action_compare(action_input)
        elif action == ActionType.LOOKUP:
            return self._action_retrieve(action_input, k)
        else:
            return "Unknown action"
    
    def _action_retrieve(self, query: str, k: int) -> str:
        """Execute retrieve action."""
        try:
            # Use retriever to search
            if hasattr(self.retriever, 'retrieve_with_context'):
                result = self.retriever.retrieve_with_context(query, k=k)
                context = result.get("context", "")
                sources = result.get("sources", [])
            else:
                results = self.retriever.search(query, k=k)
                context = "\n\n".join([r.get("content", r.get("text", "")) for r in results])
                sources = results
            
            # Store sources for later
            self._last_retrieval_sources = sources
            
            if not context:
                return "لم يتم العثور على معلومات ذات صلة في المستندات."
            
            # Truncate for observation
            return context[:1500] + ("..." if len(context) > 1500 else "")
            
        except Exception as e:
            logger.error(f"Retrieve action error: {e}")
            return f"خطأ في البحث: {str(e)}"
    
    def _action_calculate(self, expression: str) -> str:
        """Execute simple calculation."""
        try:
            # Extract numbers and operation
            numbers = re.findall(r'\d+(?:\.\d+)?', expression)
            
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                
                if '+' in expression or 'جمع' in expression:
                    result = a + b
                elif '-' in expression or 'طرح' in expression:
                    result = a - b
                elif '*' in expression or 'ضرب' in expression or '×' in expression:
                    result = a * b
                elif '/' in expression or 'قسمة' in expression or '÷' in expression:
                    result = a / b if b != 0 else "خطأ: القسمة على صفر"
                else:
                    result = a - b  # Default to subtraction for comparisons
                
                return f"النتيجة: {result}"
            
            return "لم أتمكن من استخراج الأرقام للحساب"
            
        except Exception as e:
            return f"خطأ في الحساب: {str(e)}"
    
    def _action_compare(self, expression: str) -> str:
        """Execute comparison."""
        try:
            numbers = re.findall(r'\d+(?:\.\d+)?', expression)
            
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                
                if a < b:
                    return f"{a} أقل من {b} (الفرق: {b - a})"
                elif a > b:
                    return f"{a} أكبر من {b} (الفرق: {a - b})"
                else:
                    return f"{a} يساوي {b}"
            
            return "لم أتمكن من استخراج الأرقام للمقارنة"
            
        except Exception as e:
            return f"خطأ في المقارنة: {str(e)}"
    
    def _synthesize_answer(
        self,
        query: str,
        steps: List[AgentStep],
        sources: List[Dict[str, Any]],
    ) -> str:
        """Synthesize final answer from gathered information."""
        # Collect all observations
        observations = [step.observation for step in steps if step.observation != "[Final Answer]"]
        
        combined_context = "\n\n".join(observations)
        
        # Use LLM to synthesize
        if self.llm_client:
            synthesis_prompt = f"""
بناءً على المعلومات التالية التي تم جمعها، أجب على السؤال:

السؤال: {query}

المعلومات المجمعة:
{combined_context[:3000]}

قدم إجابة موجزة ودقيقة مع ذكر المصادر.
"""
            
            try:
                result = self.llm_client.generate(
                    query=query,
                    context=combined_context,
                    sources=[s.get("source", "unknown") for s in sources[:5]],
                )
                return result.answer
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
        
        # Fallback
        if observations:
            return f"بناءً على البحث:\n\n{observations[-1][:500]}"
        
        return "لم أتمكن من إيجاد إجابة كافية."


def create_react_agent(
    retriever,
    llm_client,
    max_iterations: int = 5,
    language: str = "ar",
) -> ReActAgent:
    """Factory function to create ReAct agent."""
    return ReActAgent(
        retriever=retriever,
        llm_client=llm_client,
        max_iterations=max_iterations,
        language=language,
    )
