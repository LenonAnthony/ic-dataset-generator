import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aac_struct_gen.services import AACService
from aac_struct_gen.utils import load_environment, initialize_llm
from aac_struct_gen.models  import DatasetRow

def generate_structs(max_iterations=1):
    token = load_environment()
    llm = initialize_llm(token)

    aac_service = AACService(llm)

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
                synonym_responses = aac_service.generate_synonyms(inputs, synonym_system_prompt)
                new_inputs = [synonym for response in synonym_responses for synonym in response.synonyms]
                print(f"Generated synonyms: {new_inputs}")

            except Exception as e:
                print(f"Error generated synonyms: {e}")
                break

            try:
                card_system_prompt = """You will take the role of an expert assistant specialized in creating Augmentative and Alternative Communication (AAC) speech cards for children and individuals with various needs, including:
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
                                    Desinfetar as Mãos, eu quero desinfetar as mãos, 🧴"""  
                card_responses = aac_service.generate_cards(new_inputs, card_system_prompt)
                new_data = [DatasetRow(input=response.input, output="\n".join(response.output)) for response in card_responses]
                print(f"Generated cards: {new_data}")
                aac_service.update_dataset(output_file, new_data)
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
    max_iterations = 10
    generate_structs(max_iterations)