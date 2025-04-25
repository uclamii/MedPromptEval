from .prompt_generator import BasePromptGenerator
from .metrics_evaluator import BaseMetricsEvaluator

class CancerScreeningPromptGenerator(BasePromptGenerator):
    def get_prompt_types(self):
        return {
            "chain of thought": (
                "Encourage the chatbot to break down its reasoning into clear, logical steps. "
                "It should guide the user by explaining the thought process behind each answer, "
                "ensuring transparency and improving comprehension for complex cancer screening-related questions."
            ),
            "trigger chain of thought": (
                "Incorporate a specific trigger in the system prompt to initiate step-by-step reasoning. "
                "This approach ensures that the chatbot methodically explains cancer screening concepts by first identifying key elements "
                "before proceeding with logical deductions."
            ),
            "self consistency": (
                "Enable the chatbot to generate multiple reasoning paths and cross-check them to improve accuracy. "
                "By considering different possible answers and converging on the most consistent explanation, "
                "the chatbot will enhance reliability when answering cancer screening-related queries."
            ),
            "prompt chaining": (
                "Guide the chatbot to use a series of linked prompts to methodically process cancer screening-related questions. "
                "This ensures that responses are structured, allowing the chatbot to break down complex topics into smaller, "
                "manageable sections before arriving at a final answer."
            ),
            "ReAct": (
                "Instruct the chatbot to combine reasoning and action in an iterative process. "
                "It should analyze the user's cancer screening-related question, respond with a partial answer, "
                "evaluate the effectiveness of that response, and refine its answer accordingly."
            ),
            "tree of thoughts": (
                "Encourage the chatbot to explore multiple reasoning branches before selecting the optimal answer. "
                "This method allows for a more diverse analysis of cancer screening-related inquiries, ensuring that various angles are "
                "considered before providing a well-rounded response."
            ),
            "role based": (
                "Assign the chatbot a specific role (e.g., oncologist, screening coordinator, patient navigator) "
                "so that it can answer cancer screening-related questions with domain-specific expertise. "
                "This method ensures that responses are more authoritative and tailored to the user's needs."
            ),
            "context based": (
                "Ensure that the chatbot uses detailed context from cancer screening guidelines to inform its responses. "
                "It should extract relevant information based on user queries, ensuring that answers align "
                "with official screening protocols and procedures."
            ),
            "metacognitive prompting": (
                "Direct the chatbot to explain its reasoning process explicitly. "
                "Before providing a final answer, it should outline the key assumptions, considerations, "
                "and logical steps involved in reaching a conclusion. "
                "This enhances transparency and helps users understand the underlying rationale behind each response."
            ),
            "uncertainty-based prompting": (
                "Instruct the chatbot to acknowledge uncertainty when it lacks complete information. "
                "Rather than generating potentially misleading responses, it should indicate when additional details "
                "or verification from medical sources are needed. "
                "This improves trust and ensures responsible AI-assisted guidance."
            ),
            "guided prompting": (
                "Encourage the chatbot to ask clarifying questions before providing an answer. "
                "This ensures that responses are tailored to the user's specific needs and that "
                "the chatbot gathers enough context before delivering guidance on cancer screening topics."
            ),
        }

    def get_system_prompt_template(self):
        return """You are an expert prompt engineer. Your task is to create a system prompt for a chatbot that answers questions about cancer cascade screening using the {prompt_type} methodology: {definition}.

        Ensure that:
        1. The system prompt is clear and instructs the chatbot effectively.
        2. Responses should align with current cancer screening guidelines and protocols.
        3. Avoid unnecessary repetition.
        4. Consider the sensitive nature of cancer screening topics.

        Now, generate a system prompt for the chatbot:"""

class CancerScreeningMetricsEvaluator(BaseMetricsEvaluator):
    def get_evaluation_criteria(self):
        return "Evaluate if the answer is factually accurate and complete based on current cancer screening guidelines and protocols."

    def get_evaluation_steps(self):
        return [
            "Check if the facts in the answer are consistent with current cancer screening guidelines.",
            "Ensure that no important details about screening procedures and protocols are omitted.",
            "Prefer answers that are specific and relevant to cancer screening documentation.",
            "Make sure there are no contradictions or inconsistencies in the answer.",
            "Verify that the answer maintains appropriate sensitivity when discussing cancer-related topics.",
            "Ensure the answer includes relevant information about follow-up procedures and next steps."
        ] 