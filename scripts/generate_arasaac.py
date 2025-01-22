import json
from dataclasses import dataclass
from pathlib import Path
from typing import List
from tqdm import tqdm
import os
import sys
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aac_struct_gen.services import AACService
from aac_struct_gen.utils import load_environment, initialize_llm
from aac_struct_gen.models import DatasetRow

@dataclass
class ArasaacConfig:
    batch_size: int = 10
    output_file: str = "dataset_with_arasaac.csv"
    input_file: str = "arasaac_br.json"

class ArasaacProcessor:
    def __init__(self, config: ArasaacConfig, aac_service: AACService):
        self.config = config
        self.aac_service = aac_service

    def load_words(self) -> List[str]:
        try:
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                words = data.get("words", [])
                return [word.strip() for word in words if word.strip()]
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

    def process_batch(self, batch: List[str]) -> List[DatasetRow]:
        try:
            card_responses = self.aac_service.generate_cards(
                batch,
                self._get_card_system_prompt()
            )
            return [
                DatasetRow(input=response.input, output="\n".join(response.output))
                for response in card_responses
            ]
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

    def process_all_words(self) -> bool:
        words = self.load_words()
        if not words:
            print("Error loading words")
            return False

        total_batches = (len(words) + self.config.batch_size - 1) // self.config.batch_size
        processed_count = 0
        
        try:
            with tqdm(total=len(words), desc="Processing words") as pbar:
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * self.config.batch_size
                    end_idx = start_idx + self.config.batch_size
                    current_batch = words[start_idx:end_idx]

                    new_data = self.process_batch(current_batch)
                    if new_data:
                        self.aac_service.update_dataset(self.config.output_file, new_data)
                        processed_count += len(new_data)
                        pbar.update(len(new_data))
                        pbar.set_postfix_str(f"Processed words: {processed_count}/{len(words)}")
            
            print("Done processing all words")
            return True

        except KeyboardInterrupt:
            print(f"\nKeyboard Interrupt in: {processed_count}/{len(words)}")
            return False
        except Exception as e:
            print(f"Error: {str(e)}\n{traceback.format_exc()}")
            return False

    @staticmethod
    def _get_card_system_prompt() -> str:
        return """You will take the role of an expert assistant specialized in creating Augmentative and Alternative Communication (AAC) speech cards for children and individuals with various needs, including:
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

def main():
    try:
        config = ArasaacConfig()
        
        if not Path(config.input_file).exists():
            print(f"File {config.input_file} not found")
            return
        
        token = load_environment()
        llm = initialize_llm(token)
        aac_service = AACService(llm)
        
        processor = ArasaacProcessor(config, aac_service)
        success = processor.process_all_words()
        
        if not success:
            print("Process failed")

    except Exception as e:
        print(f"Fatal Error: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()