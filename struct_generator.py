import os
import pandas as pd
from dotenv import load_dotenv
from distilabel.llms import AzureOpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


def generate_structs(max_iterations=10):
    token = os.environ["GITHUB_TOKEN"]
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "gpt-4o-mini"
    api_version = "2023-12-01-preview"

    llm = AzureOpenAILLM(
        model=model_name,
        api_key=token,
        base_url=endpoint,
        api_version=api_version,
    )

    input_file = "dataset.csv"
    output_file = "dataset.csv"
    iteration = 0

    try:
        while iteration < max_iterations:
            print(f"\nStep {iteration + 1} / {max_iterations}")

            dataset = pd.read_csv(input_file)
            inputs = dataset["input"].sample(2).tolist()
            print(f"Selected Words: {inputs}")

            try:
                with Pipeline("SynonymGeneration") as synonym_pipeline:
                    synonym_system_prompt = """You are a specialized educational assistant focused on generating contextually relevant synonyms in Brazilian Portuguese for AAC (Augmentative and Alternative Communication) systems. Your task is to analyze the input phrase or expression and generate semantically equivalent alternatives that preserve the complete meaning and context, using natural Brazilian Portuguese expressions as commonly used today.

                                            For each input phrase:
                                            - Generate exactly 3 alternative expressions that:
                                            1. Maintain the same semantic meaning as the complete input
                                            2. Use contemporary, natural Brazilian Portuguese
                                            3. Reflect everyday speech while maintaining clarity
                                            4. Are appropriate for pictogram representation
                                            5. Are easily understood by children

                                            Example inputs and expected outputs:

                                            Input: "escovar os dentes"
                                            Expected output: limpar os dentes, passar escova nos dentes, fazer a escovação

                                            Input: "estou com fome"
                                            Expected output: quero comer, preciso comer, sinto fome

                                            Input: "quero água"
                                            Expected output: preciso beber água, quero beber água, estou com sede

                                            Input: "estou cansado"
                                            Expected output: preciso descansar, estou sem energia, estou esgotado

                                            Input: "vamos brincar"
                                            Expected output: quer brincar comigo, vamos nos divertir, vamos jogar

                                            Input: "preciso de ajuda"
                                            Expected output: pode me ajudar, preciso de auxílio, me ajude por favor

                                            Input: "não estou bem"
                                            Expected output: estou doente, me sinto mal, estou indisposto

                                            Input: "quero ir ao banheiro"
                                            Expected output: preciso ir ao banheiro, preciso usar o banheiro, quero usar o banheiro

                                            Rules for synonym generation:
                                            1. Use natural Brazilian Portuguese as commonly spoken today
                                            2. Keep expressions clear and accessible while avoiding slang
                                            3. Maintain appropriate level of formality for educational context
                                            4. Ensure expressions are suitable for all age groups
                                            5. Consider ease of pictogram representation

                                            Output format: Return only the three semantically equivalent expressions as a comma-separated list in Brazilian Portuguese, without any additional text or formatting."""

                    load_words = LoadDataFromDicts(
                        name="load_words",
                        data=[
                            {
                                "system_prompt": synonym_system_prompt,
                                "instruction": f"Generate synonyms for the word: {word}",
                            }
                            for word in inputs
                        ],
                    )

                    synonym_generation = TextGeneration(
                        name="synonym_generation",
                        llm=llm,
                        input_batch_size=8,
                        output_mappings={"model_name": "generation_model"},
                    )
                    load_words >> synonym_generation

                synonym_distiset = synonym_pipeline.run(
                    parameters={
                        synonym_generation.name: {
                            "llm": {"generation_kwargs": {"max_new_tokens": 256}}
                        }
                    },
                    use_cache=False,
                )

                synonym_results = synonym_distiset["default"]["train"]
                new_inputs = []

                for result in synonym_results:
                    if "generation" in result:
                        generated_text = result["generation"]
                        synonyms = [
                            synonym.strip() for synonym in generated_text.split(",")
                        ]
                        new_inputs.extend(synonyms)

                print(f"Generated synonyms: {new_inputs}")

            except Exception as e:
                print(f"Error generated synonyms: {e}")
                break

            try:
                with Pipeline("CardGeneration") as card_pipeline:
                    system_prompt = """You will take the role of an expert assistant specialized in creating Augmentative and Alternative Communication (AAC) speech cards for children and individuals with various needs, including:
                                    - Autism Spectrum Disorder (ASD)
                                    - Speech impediments
                                    - Motor coordination difficulties
                                    - Developmental delays
                                    - Communication disorders
                                    - Non-verbal individuals

                                    ALWAYS Format your response exactly as follows:

                                    input: [word]
                                    output: [list of options]

                                    Rules for the output:
                                    - Each option must have text, spoken_text, and an emoji, separated by commas
                                    - Generate exactly 5 options following these guidelines:
                                    * Use simple, clear, and direct language
                                    * Maintain consistent sentence structures
                                    * Use concrete rather than abstract concepts
                                    * Include common daily situations
                                    * Ensure phrases are age-appropriate
                                    * Consider motor and speech limitations
                                    * Use positive and encouraging language
                                    * Avoid complex or ambiguous expressions
                                    - Last element must be an emoji that is:
                                    * Clearly recognizable
                                    * Visually simple
                                    * Directly related to the action/object
                                    * High contrast
                                    * Commonly used
                                    - Use Brazilian Portuguese with:
                                    * Simple grammar structures
                                    * Clear pronunciation patterns
                                    * Common everyday vocabulary
                                    * Consistent verb tenses
                                    * Direct communication style
                                    - Do not include counters or extra text

                                    Focus on:
                                    - Basic needs
                                    - Daily routines
                                    - Emotional expressions
                                    - Social interactions
                                    - Emergency situations
                                    - Common requests
                                    - Personal care
                                    - Learning activities

                                    Example:
                                    input: Ação
                                    output: Abrir, eu quero abrir, 🔓
                                    Fechar, eu quero fechar, 🔒
                                    Ligar, eu quero ligar, 🔌
                                    Desligar, eu quero desligar, 🔌❌
                                    Subir, eu quero subir, ⬆️

                                    input: Banheiro
                                    output: Ir ao Banheiro, eu preciso ir ao banheiro, 🚻
                                    Pedir para Usar o Banheiro, eu gostaria de usar o banheiro, 🚽
                                    Lavar as Mãos, eu quero lavar as mãos, 🧼
                                    Buscar Papel Higiênico, eu preciso de papel higiênico, 🧻
                                    Desinfetar as Mãos, eu quero desinfetar as mãos, 🧴
                                    """

                    load_synonyms = LoadDataFromDicts(
                        name="load_synonyms",
                        data=[
                            {
                                "system_prompt": system_prompt,
                                "instruction": (
                                    f"Create a speech card following EXACTLY this format:\n"
                                    f"input: {word}\n"
                                    f"output: [5 options]"
                                ),
                            }
                            for word in new_inputs
                        ],
                    )

                    card_generation = TextGeneration(
                        name="card_generation",
                        llm=llm,
                        input_batch_size=8,
                        output_mappings={"model_name": "generation_model"},
                    )
                    load_synonyms >> card_generation

                card_distiset = card_pipeline.run(
                    parameters={
                        card_generation.name: {
                            "llm": {"generation_kwargs": {"max_new_tokens": 256}}
                        }
                    },
                    use_cache=False,
                )

                new_data = []
                for synonym, card in zip(new_inputs, card_distiset["default"]["train"]):
                    if "generation" in card:
                        generated_text = card["generation"]
                        print(f"Card generated to {synonym}:\n{generated_text}\n")
                        try:
                            parts = generated_text.split("output:")
                            if len(parts) == 2:
                                input_part = parts[0].replace("input:", "").strip()
                                output_part = parts[1].strip()
                                new_data.append(
                                    {"input": input_part, "output": output_part}
                                )
                            else:
                                new_data.append(
                                    {"input": synonym, "output": generated_text}
                                )
                        except Exception as e:
                            print(f"Error in card generation {synonym}: {e}")
                            new_data.append(
                                {"input": synonym, "output": generated_text}
                            )

                new_dataset = pd.DataFrame(new_data)
                combined_dataset = pd.concat([dataset, new_dataset], ignore_index=True)
                combined_dataset.to_csv(output_file, index=False)
                print(f"Dataset updated")

            except Exception as e:
                print(f"Error in cards pipeline: {e}")
                break

            iteration += 1

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt")
    except Exception as e:
        print(f"\nError not expected: {e}")
    finally:
        print(f"\nFinished process with {iteration} iterations")


if __name__ == "__main__":
    load_dotenv()
    max_iterations = 5
    generate_structs(max_iterations)
