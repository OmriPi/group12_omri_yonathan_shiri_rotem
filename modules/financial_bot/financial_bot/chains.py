import os
import re
import time
import openai
from typing import Any, Dict, List, Optional

import qdrant_client
from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline
from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)

from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.template import PromptTemplate

from modules.financial_bot.financial_bot import constants
from modules.financial_bot.financial_bot.openai_wrapper import OpenAIWrapper


class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    A sequential chain that uses a stateless memory to store context between calls.

    This chain overrides the _call and prep_outputs methods to load and clear the memory
    before and after each call, respectively.
    """

    history_input_key: str = "to_load_history"

    def _call(self, inputs: Dict[str, str], **kwargs) -> Dict[str, str]:
        """
        Override _call to load history before calling the chain.

        This method loads the history from the input dictionary and saves it to the
        stateless memory. It then updates the inputs dictionary with the memory values
        and removes the history input key. Finally, it calls the parent _call method
        with the updated inputs and returns the results.
        """

        to_load_history = inputs[self.history_input_key]
        for (
            human,
            ai,
        ) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """
        Override prep_outputs to clear the internal memory after each call.

        This method calls the parent prep_outputs method to get the results, then
        clears the stateless memory and removes the memory key from the results
        dictionary. It then returns the updated results.
        """

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results


class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.

    Attributes:
    -----------
    top_k : int
        The number of top matches to retrieve from the vector store.
    embedding_model : EmbeddingModelSingleton
        The embedding model to use for encoding the question.
    vector_store : qdrant_client.QdrantClient
        The vector store to search for matches.
    vector_collection : str
        The name of the collection to search in the vector store.
    """

    top_k: int = 3
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return ["context"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        _, quest_key = self.input_keys
        question_str = inputs[quest_key]

        cleaned_question = self.clean(question_str)
        # TODO: Instead of cutting the question at 'max_input_length', chunk the question in 'max_input_length' chunks,
        # pass them through the model and average the embeddings.
        cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
        embeddings = self.embedding_model(cleaned_question)

        # TODO: Using the metadata, use the filter to take into consideration only the news from the last 24 hours
        # (or other time frame).
        matches = self.vector_store.search(
            query_vector=embeddings,
            limit=self.top_k,
            collection_name=self.vector_collection,
        )

        context = ""
        for match in matches:
            context += match.payload["summary"] + "\n"

        return {
            "context": context,
        }

    def clean(self, question: str) -> str:
        """
        Clean the input question by removing unwanted characters.

        Parameters:
        -----------
        question : str
            The input question to clean.

        Returns:
        --------
        str
            The cleaned question.
        """
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question

class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""
    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        """Returns a list of input keys for the chain"""
        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        """Returns a list of output keys for the chain"""
        return ["answer"]

    def add_chain_of_thought(self) -> str:
        """Adds a chain of thought prompt to guide reasoning."""
        return (
            "Let's think step by step to provide a thorough and accurate response:"
        )

    def enrich_about_me(self, inputs) -> str:
        openai_llm = OpenAIWrapper()
        enrich_about_me_response = openai_llm.generate_response(
            prompt=f"please take the about_me field and generate more instructions based on that related to the financial bot, "
                   f"please make it short \n"
                   f"about_me={inputs['about_me']}\n")

        print(f"enrich_about_me_response: {enrich_about_me_response}")
        inputs["about_me"] += enrich_about_me_response

    def choose_best_response(self, responses: List[str]) -> str:
        """Aggregates multiple responses to find the most consistent answer."""
        if len(responses) == 1:
            return responses[0]

        openai_llm = OpenAIWrapper()
        evaluation_prompt = (
                "You are an expert assistant with financial expertise, helping to evaluate multiple answers to a question. "
                "Choose the best response based on accuracy, clarity, and relevance to the question.\n"
                "\n"
                f"Responses:\n"
                + "\n".join([f"Response {i + 1}: {response}" for i, response in enumerate(responses)]) + "\n"
                                                                                                         "\n"
                "Provide the number of the best response and explain your reasoning briefly."
        )

        evaluation_result = openai_llm.generate_response(prompt=evaluation_prompt)

        print(f"Evaluation Result: {evaluation_result}")

        # Extract the chosen response number from the evaluation result
        chosen_response_index = self.extract_chosen_response_index(evaluation_result)
        return responses[chosen_response_index]

    def extract_chosen_response_index(self, evaluation_result: str) -> int:
        """Extracts the index of the chosen response from the evaluation result."""
        match = re.search(r"Response (\d+)", evaluation_result)
        if match:
            return int(match.group(1)) - 1  # Convert to zero-based index
        else:
            raise ValueError("Failed to extract chosen response index from evaluation result.")

    def generate_multiple_responses(self, prompt: str) -> List[str]:
        """Generates multiple responses for self-consistency."""
        responses = []
        num_consistency_samples = constants.NUM_CONSISTENCY_SAMPLES
        for _ in range(num_consistency_samples):
            response = self.hf_pipeline(prompt)
            responses.append(response.strip())
        return responses


    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        use_about_me_context_enrichment: bool = True,
        use_zero_cot: bool = True,
    ) -> Dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        inputs = self.clean(inputs)

        if use_about_me_context_enrichment:
            self.enrich_about_me(inputs)

        zero_cot = ""
        if use_zero_cot:
            zero_cot = self.add_chain_of_thought()

        prompt = self.template.format_infer(
            {
                "user_context": inputs["about_me"],
                "instructions": zero_cot,
                "news_context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        full_prompt = prompt["prompt"]
        start_time = time.time()

        print (f"full_prompt: {full_prompt}")
        # Generate multiple responses
        responses = self.generate_multiple_responses(full_prompt)
        # Aggregate responses for self-consistency
        final_response = self.choose_best_response(responses)

        print (f"final_response: {final_response}")

        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if run_manager:
            run_manager.on_chain_end(
                outputs={
                    "answer": final_response,
                },
                # TODO: Count tokens instead of using len().
                metadata={
                    "prompt": full_prompt,
                    "prompt_template_variables": prompt["payload"],
                    "prompt_template": self.template.infer_raw_template,
                    "usage.prompt_tokens": len(full_prompt),
                    "usage.total_tokens": len(full_prompt) + len(final_response),
                    "usage.actual_new_tokens": len(final_response),
                    "duration_milliseconds": duration_milliseconds,
                },
            )

        return {"answer": final_response}

    def clean(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Cleans the inputs by removing extra whitespace and grouping broken paragraphs"""

        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs
