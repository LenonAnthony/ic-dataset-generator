from typing import List
import pandas as pd
from .models import (
    CardRequest,
    SynonymRequest,
    SynonymResponse,
    CardResponse,
    DatasetRow,
)
from .pipelines import SynonymPipeline, CardPipeline


class AACService:
    def __init__(self, llm):
        self.llm = llm
        self.synonym_pipeline = SynonymPipeline(llm)
        self.card_pipeline = CardPipeline(llm)

    def generate_synonyms(
        self, inputs: List[str], system_prompt: str
    ) -> List[SynonymResponse]:
        requests = [
            SynonymRequest(
                system_prompt=system_prompt,
                instruction=f"Generate synonyms for the word: {word}",
            )
            for word in inputs
        ]
        pipeline = self.synonym_pipeline.create_pipeline(requests)
        distiset = pipeline.run(
            parameters={
                "synonym_generation": {
                    "llm": {"generation_kwargs": {"max_new_tokens": 256}}
                }
            },
            use_cache=False,
        )
        results = distiset["default"]["train"]

        synonym_responses = []
        for result in results:
            if "generation" in result:
                generated_text = result["generation"]
                synonyms = [synonym.strip() for synonym in generated_text.split(",")]
                synonym_responses.append(
                    SynonymResponse(
                        input=result["instruction"].split(": ")[1], synonyms=synonyms
                    )
                )

        return synonym_responses

    def generate_cards(
        self, inputs: List[str], system_prompt: str
    ) -> List[CardResponse]:
        requests = [
            CardRequest(
                system_prompt=system_prompt,
                instruction=f"Create a speech card following EXACTLY this format:\ninput: {word}\noutput: [5 options]",
            )
            for word in inputs
        ]
        pipeline = self.card_pipeline.create_pipeline(requests)
        distiset = pipeline.run(
            parameters={
                "card_generation": {
                    "llm": {"generation_kwargs": {"max_new_tokens": 256}}
                }
            },
            use_cache=False,
        )
        results = distiset["default"]["train"]

        card_responses = []
        for result in results:
            if "generation" in result:
                generated_text = result["generation"]
                parts = generated_text.split("output:")
                if len(parts) == 2:
                    input_part = parts[0].replace("input:", "").strip()
                    output_part = parts[1].strip()
                    card_responses.append(
                        CardResponse(input=input_part, output=output_part.split("\n"))
                    )
                else:
                    card_responses.append(
                        CardResponse(
                            input=result["instruction"].split(": ")[1],
                            output=generated_text.split("\n"),
                        )
                    )

        return card_responses

    def update_dataset(self, dataset_path: str, new_data: List[DatasetRow]):
        dataset = pd.read_csv(dataset_path)
        new_dataset = pd.DataFrame([row.model_dump() for row in new_data])
        combined_dataset = pd.concat([dataset, new_dataset], ignore_index=True)
        combined_dataset.to_csv(dataset_path, index=False)
