from typing import List
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from .models import SynonymRequest, CardRequest


class SynonymPipeline:
    def __init__(self, llm):
        self.llm = llm

    def create_pipeline(self, requests: List[SynonymRequest]):
        with Pipeline("SynonymGeneration") as pipeline:
            load_words = LoadDataFromDicts(
                name="load_words",
                data=[request.model_dump() for request in requests],
            )

            synonym_generation = TextGeneration(
                name="synonym_generation",
                llm=self.llm,
                input_batch_size=8,
                output_mappings={"model_name": "generation_model"},
            )

            load_words >> synonym_generation

        return pipeline


class CardPipeline:
    def __init__(self, llm):
        self.llm = llm

    def create_pipeline(self, requests: List[CardRequest]):
        with Pipeline("CardGeneration") as pipeline:
            load_synonyms = LoadDataFromDicts(
                name="load_synonyms",
                data=[request.model_dump() for request in requests],
            )

            card_generation = TextGeneration(
                name="card_generation",
                llm=self.llm,
                input_batch_size=8,
                output_mappings={"model_name": "generation_model"},
            )

            load_synonyms >> card_generation

        return pipeline
