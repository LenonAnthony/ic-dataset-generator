# IC Dataset Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Build Status](https://github.com/LenonAnthony/ic-dataset-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/LenonAnthony/ic-dataset-generator/actions)

## Sobre

O **IC Dataset Generator** é um código implementado para auxiliar no design e implementação de pequenos modelos de linguagem otimizados para Comunicação Aumentativa e Alternativa (AAC). Este projeto utiliza o poder dos modelos de linguagem da Azure OpenAI e a biblioteca aberta Distilabel para gerar sinônimos e cartões de fala em português brasileiro, visando melhorar a comunicação de indivíduos com deficiências de comunicação.

## Funcionalidades

- **Geração de Sinônimos**: Através do distilabel, cria um pipeline junto a modelos de linguagem para gerar sinônimos contextualmente apropriados em português brasileiro.
- **Criação de Cartões de Fala**: Gera textos organizados que podem ser convertidos em cartões de fala formatados com texto, texto falado e emojis, para uso futuro em ferramentas de comunicação AAC.
- **Integração com Azure OpenAI**: Utiliza a API da Azure OpenAI para geração de texto.

## Requisitos

- Python 3.8+
- Conta na Azure com acesso ao Azure OpenAI

## Instalação

Clone o repositório e instale as dependências:

```sh
git clone https://github.com/LenonAnthony/ic-dataset-generator.git
cd ic-dataset-generator
pip install -r requirements.txt
```

## Configuração

Crie um arquivo .env na raiz do projeto e adicione suas variáveis de ambiente: GITHUB_TOKEN=your_github_token ou your_azure_api_token

## Uso 

python dataset_generator.py



Este projeto foi desenvolvido como parte de um artigo sobre o design e implementação de modelos de linguagem pequenos para Comunicação Aumentativa e Alternativa (AAC). Para mais informações, consulte o artigo completo
(colocar depois).


