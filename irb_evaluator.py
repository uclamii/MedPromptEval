from .prompt_generator import BasePromptGenerator
from .metrics_evaluator import BaseMetricsEvaluator

class IRBPromptGenerator(BasePromptGenerator):
    def get_prompt_types(self):
        return {
            "chain of thought": (
                "Encourage the chatbot to break down its reasoning into clear, logical steps. "
                "It should guide the user by explaining the thought process behind each answer, "
                "ensuring transparency and improving comprehension for complex IRB-related questions."
            ),
            "trigger chain of thought": (
                "Incorporate a specific trigger in the system prompt to initiate step-by-step reasoning. "
                "This approach ensures that the chatbot methodically explains IRB-related concepts by first identifying key elements "
                "before proceeding with logical deductions."
            ),
            "self consistency": (
                "Enable the chatbot to generate multiple reasoning paths and cross-check them to improve accuracy. "
                "By considering different possible answers and converging on the most consistent explanation, "
                "the chatbot will enhance reliability when answering IRB-related queries."
            ),
            "prompt chaining": (
                "Guide the chatbot to use a series of linked prompts to methodically process IRB-related questions. "
                "This ensures that responses are structured, allowing the chatbot to break down complex topics into smaller, "
                "manageable sections before arriving at a final answer."
            ),
            "ReAct": (
                "Instruct the chatbot to combine reasoning and action in an iterative process. "
                "It should analyze the user's IRB-related question, respond with a partial answer, "
                "evaluate the effectiveness of that response, and refine its answer accordingly."
            ),
            "tree of thoughts": (
                "Encourage the chatbot to explore multiple reasoning branches before selecting the optimal answer. "
                "This method allows for a more diverse analysis of IRB-related inquiries, ensuring that various angles are "
                "considered before providing a well-rounded response."
            ),
            "role based": (
                "Assign the chatbot a specific role (e.g., IRB expert, compliance officer, research ethics advisor) "
                "so that it can answer IRB-related questions with domain-specific expertise. "
                "This method ensures that responses are more authoritative and tailored to the user's needs."
            ),
            "context based": (
                "Ensure that the chatbot uses detailed context from IRB documentation to inform its responses. "
                "It should extract relevant information based on user queries, ensuring that answers align "
                "with official IRB policies and procedures."
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
                "or verification from official IRB sources are needed. "
                "This improves trust and ensures responsible AI-assisted guidance."
            ),
            "guided prompting": (
                "Encourage the chatbot to ask clarifying questions before providing an answer. "
                "This ensures that responses are tailored to the user's specific needs and that "
                "the chatbot gathers enough context before delivering guidance on IRB-related topics."
            ),
        }

    def get_system_prompt_template(self):
        return """You are an expert prompt engineer. Your task is to create a system prompt for a chatbot that answers questions about IRB documentation using the {prompt_type} methodology: {definition}.

        Ensure that:
        1. The system prompt is clear and instructs the chatbot effectively.
        2. Responses should align with the IRB documentation.
        3. Avoid unnecessary repetition.

        Now, generate a system prompt for the chatbot:"""

class IRBMetricsEvaluator(BaseMetricsEvaluator):
    def get_evaluation_criteria(self):
        return "Evaluate if the answer is factually accurate and complete based on IRB documentation."

    def get_evaluation_steps(self):
        return [
            "Check if the facts in the answer are consistent with known IRB procedures and guidelines.",
            "Ensure that no important details about IRB processes are omitted.",
            "Prefer answers that are specific and relevant to IRB documentation.",
            "Make sure there are no contradictions or inconsistencies in the answer."
        ] 