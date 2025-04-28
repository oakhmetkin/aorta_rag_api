from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import networkx as nx
import re
import unicodedata
import logging
import os

from .generative_model import GenerativeModel


__all__ = ['QwenModel', 'QwenModelConfig']

logger = logging.getLogger('rag_logger')

# choose device
device = torch.device('cuda:1')


with open('/home/ahmetkin/aorta_rag_api/data/static_text_ru_002.txt') as f:
    STATIC_CONTEXT_RU = f.read()


with open('/home/ahmetkin/aorta_rag_api/data/static_text_en_001.txt') as f:
    STATIC_CONTEXT_EN = f.read()


# system prompts
system_prompt = {
    'ru': "Ты хорошо знаешь русский язык. Помоги решить задачу.",
    'en': 'You know english well. Help me solve the problem.',
}

# (final) answer prompts
ANSWER_PROMPT_RU = """
Ты - помощник кардиохирурга. Пожалуйста, посмотри на текст (в тройных обратных кавычках) и симптомы больного ниже.
Опираясь на факты из текста, напиши заключение, в которое будет содержать описание лечения и рекомендации для больного.
Если пациент здоров (т. е. диаметр аорты точно попадает в интервал нормы), то напиши об этом.
Твоя задача - упростить и ускорить работу кардиохирурга, поэтому напиши заключение так, как написал бы его кардиохирург.
```
{context}
```
Симптомы:
{simptoms}
Заключение:
"""

ANSWER_PROMPT_EN = """
You are a cardiac surgeon's assistant. Analyze the text below (enclosed in triple backticks) and the patient's symptoms.  
Generate a **concise clinical summary** with:  
1. **Treatment plan**: Based on factual data (e.g., aortic diameter, pathology).  
2. **Recommendations**: Actionable steps (medication, surgery, monitoring).  
3. **Normal findings**: Explicitly state if the aorta is within normal limits.  

**Style**: Write as a senior cardiothoracic surgeon would—*authoritative, precise, and clinically streamlined*.  

**Rules**:  
- **Aorta-first priority**: Highlight aortic dimensions (e.g., "4.5cm ascending aorta → surveillance").  
- **Symptom correlation**: Link symptoms to interventions (e.g., "chest pain + dissection → emergent repair").  
- **No speculation**: Only use data from the text/symptoms.  

```  
{context}  
```  

**Symptoms**:  
{simptoms}  

**Clinical Summary**:  
"""

# entity extraction prompts
ENTITY_LOOKUP_PROMPT_RU = """
Ниже в тройных обратных кавычках приводится короткий текст. Тебе необходимо выделить из него все сущности,
похожие на сущности из списка в двойных кавычках: "{list}". Верни только список сущностей в скобках
через запятую, например: (аневризма, АБА, диаметр, операция). Верни те сущности, которые в явном виде
присутствуют в запросе (они могут быть написаны с опечатками или в другом падеже),
а также те сущности, которые могли бы быть полезны для диагностики аорты (например, аневризмы).
Не придумывай никакие дополнительные сущности и не рассуждай.
--текст--
```
{}
```
"""

ENTITY_LOOKUP_PROMPT_EN = """
Analyze the text below (triple backticks) and extract ALL entities matching the list: "{list}". 
Prioritize aortic-related terms even if not explicitly listed but clinically relevant for aortic diagnosis.

Output EXACTLY in this format:
(term1, term2, term3)

Critical rules:
1. Mandatory inclusions:
   - Direct matches from [{list}] (ignore case/typos: "Aneuryzm" → "aneurysm")
   - Implied aortic terms (e.g., "root dilation" → include even if only "dilation" is listed)
   - Diagnostic markers (e.g., "CT", "echo", "blood pressure" if context suggests aortic evaluation)

2. Strict exclusions:
   - No unmentioned entities
   - No explanatory text

3. Aortic priority terms (always include):
   - Anatomic: root/ascending/arch/descending, diameter, valve
   - Pathologic: dissection/aneurysm/rupture/coarctation
   - Diagnostic: imaging modalities, pressure gradients

--text--
```
{}
```
"""


class QwenModelConfig:
    
    def __init__(self, er_ru_file: str, er_en_file: str):
        self.__ACCENT_MAPPING = {
            '́': '', '̀': '', 'а́': 'а', 'а̀': 'а', 'е́': 'е', 'ѐ': 'е', 'и́': 'и',
            'ѝ': 'и', 'о́': 'о', 'о̀': 'о', 'у́': 'у', 'у̀': 'у', 'ы́': 'ы',
            'ы̀': 'ы', 'э́': 'э', 'э̀': 'э', 'ю́': 'ю', '̀ю': 'ю', 'я́́': 'я',
            'я̀': 'я',
        }

        self.__ACCENT_MAPPING = {
            unicodedata.normalize('NFKC', i): j for i, j in self.__ACCENT_MAPPING.items()
        }

        self.entities_ru, self.relations_ru = self.__extract_ER_from_file(er_ru_file)
        self.entities_en, self.relations_en = self.__extract_ER_from_file(er_en_file)

        self.entity_lookup_prompt_ru = ENTITY_LOOKUP_PROMPT_RU.replace(
            '{list}', ', '.join(self.entities_ru.keys()),
        )

        self.entity_lookup_prompt_en = ENTITY_LOOKUP_PROMPT_EN.replace(
            '{list}', ', '.join(self.entities_en.keys()),
        )
    
    def __add_entity(self, entities, name, kind, desc):
        if name in entities.keys():
            entities[name]['kind'].append(kind)
            entities[name]['desc'].append(desc)
        else:
            entities[name] = {'kind': [kind], 'desc': [desc]}

    def __unaccentify(self, s):
        source = unicodedata.normalize('NFKC', s)
        for old, new in self.__ACCENT_MAPPING.items():
            source = source.replace(old, new)
        return source

    def __normalize(self, text):
        return (
            self.__unaccentify(text)
        )

    def __extract_ER(self, lines):
        entities = {}
        relations = []
        for x in lines:
            x = self.__normalize(x)

            if z := re.match(r'\((.*)\)', x):
                z = z.string.strip()[1:-1].split('|')
                z = [t.strip().lower() for t in z]

                if z[0] == 'entity':
                    if len(z) < 4:
                        z.append('')
                    else:
                        self.__add_entity(entities, z[1], z[2], z[3])
                elif z[0] == 'relationship':
                    while len(z) < 5:
                        z.append('')
                    relations.append({
                        "source": z[1],
                        "target": z[2],
                        "relation": z[3],
                        "desc": z[4],
                    })
                else:
                    logger.warning(f'Invalid command: {z}')

        # Clean up relations with non-existing entities
        relations = [
            x for x in relations
            if x['source'] in entities.keys() and x['target'] in entities.keys()
        ]

        return entities, relations


    def __extract_ER_from_file(self, er_file: str):
        with open(er_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        entities, relations = self.__extract_ER(lines)
        logger.info(f"Found {len(entities)} entities and {len(relations)} relations")
        return entities, relations


class QwenModel(GenerativeModel):
    
    def __init__(self, config: QwenModelConfig):
        super().__init__()


        model_name = "Qwen/Qwen2.5-32B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.__entities = {
            'ru': config.entities_ru,
            'en': config.entities_en,
        }

        self.__relations = {
            'ru': config.relations_ru,
            'en': config.relations_en,
        }

        self.__entity_lookup_prompt = {
            'ru': config.entity_lookup_prompt_ru,
            'en': config.entity_lookup_prompt_en,
        }
    
    def __generate_text(self, prompt: str, max_len: int = 512, lang: str = 'ru'):
        messages = [
            {"role": "system", "content": system_prompt[lang]},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def generate(self, message: str, max_len: int, lang: str = 'ru', *args, **kwargs):
        # ents = self.__process_q(message, lang)
        # logger.debug(f'lang="{lang}"; parsed_entities={ents}')

        # G = nx.DiGraph()
        # for e in ents:
        #     self.__populate_graph(G, e, 2)

        if lang == 'en':
            # context = self.__create_context(G)
            context = STATIC_CONTEXT_EN
        else:
            context = STATIC_CONTEXT_RU

        answer_prompt = (
            (ANSWER_PROMPT_RU if lang == 'ru' else ANSWER_PROMPT_EN)
            .replace('{context}', context)
            .replace('{simptoms}', message)
        )

        ans = self.__generate_text(
            answer_prompt,
            max_len,
            lang,
        )

        return ans
    
    def __process_q(self, txt, lang: str = 'ru'):       
        res = self.__generate_text(self.__entity_lookup_prompt[lang].format(txt))

        if '(' in res and ')' in res:
            res = res[res.index('(')+1:res.index(')')]
            res = res.split(',')
            return [x.strip() for x in res]
        else:
            return None

    def __create_context(self, G):
        return '\n'.join(
            f"{e[-1]['label']}: {e[-1]['desc']}" for e in G.edges(data=True)
        )

    def __populate_graph(self, G, e, level=None, lang: str = 'ru'):
        if e in G.nodes:
            return

        if e in self.__entities[lang].keys():
            G.add_node(e, label=e)

        if level is not None and level <= 0:
            return

        new_ent = set(
            [r['source'] for r in self.__relations[lang] if r['target'] == e] +
            [r['target'] for r in self.__relations[lang] if r['source'] == e]
        )

        for ne in new_ent:
            self.__populate_graph(G, ne, None if level is None else level-1)

        for r in self.__relations[lang]:
            if r['source'] == e:
                G.add_edge(e, r['target'], label=f"{r['target']} {r['relation']}", desc=r['desc'])
            if r['target'] == e:
                G.add_edge(r['source'], e, label=f"{r['source']} {r['relation']}", desc=r['desc'])
