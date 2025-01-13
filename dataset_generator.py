import os
import pandas as pd
from dotenv import load_dotenv
from distilabel.llms import AzureOpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


def main():
    # Carregar variáveis de ambiente
    load_dotenv()

    # Configurações
    token = os.environ["GITHUB_TOKEN"]
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "gpt-4o"
    api_version = "2023-12-01-preview"

    # Inicializar o LLM
    llm = AzureOpenAILLM(
        model=model_name,
        api_key=token,
        base_url=endpoint,
        api_version=api_version,
    )

    # Carregar o dataset
    input_file = "dataset.csv"
    output_file = "output_dataset.csv"
    dataset = pd.read_csv(input_file)

    # Extraindo inputs para gerar sinônimos
    inputs = dataset["input"].sample(1).tolist()

    # Pipeline para gerar sinônimos
    with Pipeline("SynonymGeneration") as synonym_pipeline:
        synonym_system_prompt = """You are an empathetic assistant dedicated to supporting individuals with communication disabilities. 
                                    Your task is to generate synonyms in Brazilian Portuguese for important words or phrases that these individuals might use in their daily communication. 

                                    - Generate 1 relevant and meaningful synonyms for each input word or phrase. 
                                    - Ensure that the synonyms are contextually appropriate, simple to understand, and practical for use in communication tools. 
                                    - Return the synonyms as a comma-separated list, without any additional text or formatting.

                                    Remember that your goal is to provide useful and accessible alternatives to improve communication. Consider the everyday needs of users when generating these synonyms."""

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
            synonyms = [synonym.strip() for synonym in generated_text.split(",")]
            new_inputs.extend(synonyms)

    print("Novos inputs gerados:", new_inputs)

    # Pipeline para gerar cartões
    with Pipeline("CardGeneration") as card_pipeline:
        system_prompt = """You will take the role of assistant to a creator of speech cards for people with speech disabilities.
                            ALWAYS Format your response exactly as follows:

                            input: [word]
                            output: [list of options]

                            Rules for the output:
                            - Each option must have text, spoken_text, and an emoji, separated by commas
                            - Generate exactly 5 options
                            - Last element must be an emoji
                            - Use Brazilian Portuguese
                            - Do not include counters or extra text

                            Example:
                            input: Ação
                            output: Abrir, eu quero abrir, 🔓
                            Fechar, eu quero fechar, 🔒
                            Ligar, eu quero ligar, 🔌
                            Desligar, eu quero desligar, 🔌❌
                            Subir, eu quero subir, ⬆️
                            
                            input: Ajuda
                            output: Preciso de Ajuda, eu preciso de ajuda, 🤲
                            Me Dê uma Mão, eu gostaria que você me ajudasse, 🖐️
                            Ajudar com a Tarefa, eu preciso de ajuda com a tarefa, 📝
                            Encontre Alguém para Ajudar, eu quero encontrar alguém para ajudar, 🔍
                            Ajuda Imediata, eu preciso de ajuda imediata, 🚨
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

    # Processar e adicionar resultados ao dataset
    new_data = []
    for synonym, card in zip(new_inputs, card_distiset["default"]["train"]):
        if "generation" in card:
            generated_text = card["generation"]
            print(f"texto gerado: {generated_text}")
            try:
                # Separar input e output
                parts = generated_text.split("output:")
                if len(parts) == 2:
                    input_part = parts[0].replace("input:", "").strip()
                    output_part = parts[1].strip()
                    new_data.append({"input": input_part, "output": output_part})
                else:
                    new_data.append({"input": synonym, "output": generated_text})
            except Exception as e:
                print(f"Erro ao processar card para {synonym}: {e}")
                new_data.append({"input": synonym, "output": generated_text})

    print("Novos dados gerados:", new_data)
    new_dataset = pd.DataFrame(new_data)
    combined_dataset = pd.concat([dataset, new_dataset], ignore_index=True)

    # Salvar o dataset atualizado
    combined_dataset.to_csv(output_file, index=False)
    print(f"Dataset atualizado salvo em {output_file}")


if __name__ == "__main__":
    main()
