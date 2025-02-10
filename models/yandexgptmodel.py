import os
from yandex_chain import YandexLLM, YandexGPTModel
import networkx as nx
import re
import unicodedata
import logging

from .generative_model import GenerativeModel


__all__ = ['YandexGptModel', 'YandexGptModelConfig']

logger = logging.getLogger('rag_logger')


ANSWER_PROMPT = """
Ты - помощник кардиохирурга. Пожалуйста, посмотри на текст (в тройных обратных кавычках) и симптомы больного ниже.
Опираясь на факты из текста, напиши заключение, в которое будет содержать описание лечения и рекомендации для больного.
Твоя задача - упростить и ускорить работу кардиохирурга, поэтому напиши заключение так, как написал бы его кардиохирург.
```
{context}
```
Симптомы:
{simptoms}
Заключение:
"""


ENTITY_LOOKUP_PROMPT = """
Ниже в тройных обратных кавычках приводится короткий текст. Тебе необходимо выделить из него все сущности,
похожие на сущности из списка в двойных кавычках: "{list}". Верни только список сущностей в скобках
через запятую, например: (аневризма, АБА, диаметр, операция). Верни только те сущности, которые в явном виде
присутствуют в запросе (они могут быть написаны с опечатками или в другом падеже).
Не придумывай никакие дополнительные сущности и не рассуждай.
--текст--
```
{}
```
"""


class YandexGptModelConfig:
    
    def __init__(self, er_file: str):
        self.__ACCENT_MAPPING = {
            '́': '', '̀': '', 'а́': 'а', 'а̀': 'а', 'е́': 'е', 'ѐ': 'е', 'и́': 'и',
            'ѝ': 'и', 'о́': 'о', 'о̀': 'о', 'у́': 'у', 'у̀': 'у', 'ы́': 'ы',
            'ы̀': 'ы', 'э́': 'э', 'э̀': 'э', 'ю́': 'ю', '̀ю': 'ю', 'я́́': 'я',
            'я̀': 'я',
        }

        self.__ACCENT_MAPPING = {
            unicodedata.normalize('NFKC', i): j for i, j in self.__ACCENT_MAPPING.items()
        }

        # -----

        self.folder_id = os.environ['YANDEX_FOLDER_ID']
        self.api_key = os.environ['YANDEX_API_KEY']

        self.entities, self.relations = self.__extract_ER_from_file(er_file)

        self.entity_lookup_prompt = ENTITY_LOOKUP_PROMPT.replace(
            '{list}', ', '.join(self.entities.keys()),
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
            .replace('«', '')
            .replace('»', '')
            .replace('"', '')
            .replace('<', '')
            .replace('>', '')
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
            lines = f.read()
        
        entities, relations = self.__extract_ER(lines)
        logger.info(f"Found {len(entities)} entities and {len(relations)} relations")
        return entities, relations


class YandexGptModel(GenerativeModel):
    
    def __init__(self, config: YandexGptModelConfig):
        super().__init__()

        self.__folder_id = config.folder_id
        self.__api_key = config.api_key

        self.llm = YandexLLM(
            folder_id=self.__folder_id,
            api_key=self.__api_key,
            model=YandexGPTModel.Pro,
        )

        self.__entities = config.entities
        self.__relations = config.relations

        self.__entity_lookup_prompt = config.entity_lookup_prompt

    def generate(self, message: str, max_len: int):
        ents = self.__process_q(message)

        G = nx.DiGraph()
        for e in ents:
            self.__populate_graph(G, e, 2)

        ans = self.llm.invoke(
            ANSWER_PROMPT
            .replace('{context}', self.__create_context(G))
            .replace('{simptoms}', message),
        )

        return ans
    
    def __process_q(self, txt):       
        res = self.llm.invoke(self.__entity_lookup_prompt.format(txt))

        if '(' in res and ')' in res:
            res = res[res.index('(')+1:res.index(')')]
            res = res.split(',')
            return [x.strip() for x in res]
        else:
            return None

    def __create_context(self, G):
        return '\n'.join(
            e[-1]['desc'] for e in G.edges(data=True)
        )

    def __populate_graph(self, G, e, level=None):
        if e in G.nodes:
            return

        if e in self.__entities.keys():
            G.add_node(e, label=e)

        if level is not None and level <= 0:
            return

        new_ent = set(
            [r['source'] for r in self.__relations if r['target'] == e] +
            [r['target'] for r in self.__relations if r['source'] == e]
        )

        for ne in new_ent:
            self.__populate_graph(G, ne, None if level is None else level-1)

        for r in self.__relations:
            if r['source'] == e:
                G.add_edge(e, r['target'], label=r['relation'], desc=r['desc'])
            if r['target'] == e:
                G.add_edge(r['source'], e, label=r['relation'], desc=r['desc'])
