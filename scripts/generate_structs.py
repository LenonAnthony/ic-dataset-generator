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
    used_synonyms = set() 
    current_batch_synonyms = [] 

    try:
        while iteration < max_iterations:
            print(f"\nStep {iteration + 1} / {max_iterations}")

            dataset = pd.read_csv(input_file)
            inputs = dataset["input"].sample(2).tolist()
            print(f"Selected Words: {inputs}")

            try:
                synonym_system_prompt = f"""You are a specialized educational assistant focused on generating contextually relevant synonyms and helpful words in Brazilian Portuguese for AAC (Augmentative and Alternative Communication) systems. Your task is to analyze the input phrase or expression and generate:
                                        - 2 semantically equivalent alternatives that preserve the complete meaning and context
                                        - 1 random but contextually helpful Brazilian Portuguese word that could assist children with mobility challenges, AAC users, neurodivergent individuals, etc.


                                        Critical Rule:
                                        - NEVER repeat these already generated synonyms: {', '.join(used_synonyms) if used_synonyms else 'None'}
                                        - Only generate completely new variations not in the list above
                                        
                                        For each input phrase:
                                        Generate outputs that:
                                        1. For the synonyms:
                                        - Maintain the same semantic meaning as the complete input
                                        - Use contemporary, natural Brazilian Portuguese
                                        - Reflect everyday speech while maintaining clarity
                                        - Are appropriate for pictogram representation
                                        - Are easily understood by children

                                        2. For the random word:
                                        - Is related to the context or situation
                                        - Could be helpful for communication in similar scenarios
                                        - Is simple and clear
                                        - Is easy to represent visually
                                        - Could expand the child's communication options

                                        Example inputs and expected outputs:
                                        Input: "escovar os dentes"
                                        Output: limpar os dentes, fazer a escovação, pasta de dente

                                        Input: "estou com fome"
                                        Output: quero comer, preciso comer, colher

                                        Input: "quero água"
                                        Output: preciso beber água, estou com sede, copo

                                        Input: "estou cansado"
                                        Output: preciso descansar, estou sem energia, cama

                                        Input: "vamos brincar"
                                        Output: quer brincar comigo, vamos nos divertir, bola

                                        Input: "preciso de ajuda"
                                        Output: pode me ajudar, me ajude por favor, mamãe

                                        Input: "não estou bem"
                                        Output: estou doente, me sinto mal, remédio

                                        Input: "quero ir ao banheiro"
                                        Output: preciso ir ao banheiro, quero usar o banheiro, papel

                                        Rules for generation:
                                        1. Use natural Brazilian Portuguese as commonly spoken today
                                        2. Keep expressions clear and accessible while avoiding slang
                                        3. Maintain appropriate level of formality for educational context
                                        4. Ensure expressions are suitable for all age groups
                                        5. Consider ease of pictogram representation
                                        6. The random word should be useful for expanding communication options

                                        Output format: Return only the two semantically equivalent expressions and one random helpful word as a comma-separated list in Brazilian Portuguese, without any additional text or formatting."""  
                synonym_responses = aac_service.generate_synonyms(inputs, synonym_system_prompt)
                new_synonyms = []
                for response in synonym_responses:
                    for synonym in response.synonyms:
                        if synonym not in used_synonyms:
                            new_synonyms.append(synonym)
                            used_synonyms.add(synonym)
                
                current_batch_synonyms = new_synonyms.copy()
                print(f"Novos sinônimos gerados: {current_batch_synonyms}")


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
                                    Desinfetar as Mãos, eu quero desinfetar as mãos, 🧴
                                    
                                    input: Inseto
                                    output: Qual é o nome deste inseto?, como se chama este inseto? 🐞
                                    Onde os insetos costumam viver?, onde os insetos normalmente se escondem? 🌿
                                    Como os insetos se reproduzem?, como acontece a reprodução dos insetos? 🐜
                                    Quais insetos são benéficos?, quais insetos são bons para o meio ambiente? 🌼
                                    Por que os insetos são importantes?, por que os insetos são importantes para a natureza? 🌍
                                    """  
                card_responses = aac_service.generate_cards(current_batch_synonyms, card_system_prompt)
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
    max_iterations = 150
    generate_structs(max_iterations)