import os
import json
import glob
import random
from lxml import etree
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List
from nltk import sent_tokenize_, wordpunct_tokenize_
from transformers import BertTokenizer



ROOTDIR = '/shared/nas/data/m1/yinglin8/projects/oneie/data/ere/raw/'

ERE_V1 = 'LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V1'
ERE_V2 = 'LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2'
ERE_R2_V2 = 'LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2'
ERE_PARL_V2 = 'LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2'
SPANISH = 'LDC2015E107_DEFT_Rich_ERE_Spanish_Annotation_V2'

relation_type_mapping = {
    'orgaffiliation': 'ORG-AFF',
    'personalsocial': 'PER-SOC',
    'physical': 'PHYS',
    'generalaffiliation': 'GEN-AFF',
    'partwhole': 'PART-WHOLE',
}

event_type_mapping = {
    'business:declarebankruptcy': 'Business:Declare-Bankruptcy',
    'business:endorg': 'Business:End-Org',
    'business:mergeorg': 'Business:Merge-Org',
    'business:startorg': 'Business:Start-Org',
    'conflict:attack': 'Conflict:Attack',
    'conflict:demonstrate': 'Conflict:Demonstrate',
    'contact:broadcast': 'Contact:Broadcast',
    'contact:contact': 'Contact:Contact',
    'contact:correspondence': 'Contact:Correspondence',
    'contact:meet': 'Contact:Meet',
    'justice:acquit': 'Justice:Acquit',
    'justice:appeal': 'Justice:Appeal',
    'justice:arrestjail': 'Justice:Arrest-Jail',
    'justice:chargeindict': 'Justice:Charge-Indict',
    'justice:convict': 'Justice:Convict',
    'justice:execute': 'Justice:Execute',
    'justice:extradite': 'Justice:Extradite',
    'justice:fine': 'Justice:Fine',
    'justice:pardon': 'Justice:Pardon',
    'justice:releaseparole': 'Justice:Release-Parole',
    'justice:sentence': 'Justice:Sentence',
    'justice:sue': 'Justice:Sue',
    'justice:trialhearing': 'Justice:Trial-Hearing',
    'life:beborn': 'Life:Be-Born',
    'life:die': 'Life:Die',
    'life:divorce': 'Life:Divorce',
    'life:injure': 'Life:Injure',
    'life:marry': 'Life:Marry',
    'manufacture:artifact': 'Manufacture:Artifact',
    'movement:transportartifact': 'Movement:Transport',
    'movement:transportperson': 'Movement:Transport-Person',
    'personnel:elect': 'Personnel:Elect',
    'personnel:endposition': 'Personnel:End-Position',
    'personnel:nominate': 'Personnel:Nominate',
    'personnel:startposition': 'Personnel:Start-Position',
    'transaction:transaction': 'Transaction:Transaction',
    'transaction:transfermoney': 'Transaction:Transfer-Money',
    'transaction:transferownership': 'Transaction:Transfer-Ownership',
}

role_type_mapping = {
    'victim': 'Victim',
    'attacker': 'Attacker',
    'person': 'Person',
    'plaintiff': 'Plaintiff',
    'audience': 'Audience',
    'destination': 'Destination',
    'prosecutor': 'Prosecutor',
    'target': 'Target',
    'origin': 'Origin',
    'recipient': 'Recipient',
    'beneficiary': 'Beneficiary',
    'adjudicator': 'Adjudicator',
    'thing': 'Thing',
    'giver': 'Giver',
    'defendant': 'Defendant',
    'entity': 'Entity',
    'org': 'Org',
    'agent': 'Agent',
    'place': 'Place',
    'artifact': 'Artifact',
    'instrument': 'Instrument'
}


def mask_escape(text):
    return text.replace('&amp;', 'ҪҪҪҪҪ').replace('&lt;', 'ҚҚҚҚ').replace('&gt;', 'ҺҺҺҺ')


def unmask_escape(text):
    return text.replace('ҪҪҪҪҪ', '&amp;').replace('ҚҚҚҚ', '&lt;').replace('ҺҺҺҺ', '&gt;')


def recover_escape(text):
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')


def sent_tokenize(text, language='english'):
    if language == 'chinese':
        return split_chinese_sentence(text)
    return sent_tokenize_(text, language=language)


def wordpunct_tokenize(text, language='english'):
    if language == 'chinese':
        return [c for c in text]
    return wordpunct_tokenize_(text)


def split_chinese_sentence(text):
    sentences = []
    quote_mark_count = 0
    sentence = ''
    for i, c in enumerate(text):
        sentence += c
        if c in {'”', '」'}:
            sentences.append(sentence)
            sentence = ''
        elif c in {'。', '!', '?', '！', '？'}:
            if i < len(text) - 1 and text[i + 1] not in {'”', '"', '」'}:
                sentences.append(sentence)
                sentence = ''
        elif c == '"':
            quote_mark_count += 1
            if quote_mark_count % 2 == 0 and len(sentence) > 2 and sentence[-2] in {'？', '！', '。', '?', '!'}:
                sentences.append(sentence)
                sentence = ''
    if sentence:
        sentences.append(sentence)
    return sentences


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace('\n', ' ')

    def char_offsets_to_token_offsets(self, tokens):
        start_ = end_ = -1
        for i, (s, e, _) in enumerate(tokens):
            if s == self.start:
                start_ = i
            if e == self.end:
                end_ = i + 1
        if start_ == -1 or end_ == -1 or start_ > end_:
            raise ValueError('Failed to update offsets for {}-{}:{} in {}'.format(
                self.start, self.end, self.text, tokens))
        self.start, self.end = start_, end_

    def to_dict(self):
        return {
            'text': recover_escape(self.text),
            'start': self.start,
            'end': self.end
        }

    def remove_space(self):
        # heading spaces
        text = self.text.lstrip(' ')
        self.start += len(self.text) - len(text)
        # trailing spaces
        text = text.rstrip(' ')
        self.text = text
        self.end = self.start + len(text)


@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    mention_type: str

    def to_dict(self, sent_id=None):
        if sent_id:
            entity_id = '{}-{}-{}'.format(sent_id,
                                          self.entity_id.split('-')[-1],
                                          self.mention_id.split('-')[-1])
        else:
            entity_id = '{}-{}'.format(self.entity_id.split('-')[-1],
                                       self.mention_id.split('-')[-1])
        return {
            'entity_id': entity_id,
            'entity_type': self.entity_type,
            'mention_type': self.mention_type,
            'start': self.start, 
            'end': self.end,
            'text': recover_escape(self.text)
        }


@dataclass
class RelationArgument:
    entity_id: str
    mention_id: str
    role: str
    text: str

    def to_dict(self, sent_id=None):
        if sent_id:
            entity_id = '{}-{}-{}'.format(sent_id,
                                          self.entity_id.split('-')[-1],
                                          self.mention_id.split('-')[-1])
        else:
            entity_id = '{}-{}'.format(self.entity_id.split('-')[-1],
                                       self.mention_id.split('-')[-1])

        return {
            'entity_id': entity_id,
            'role': self.role,
            'text': recover_escape(self.text)
        }


@dataclass
class Relation:
    relation_id: str
    mention_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self, sent_id=None):
        if sent_id:
            relation_id = '{}-{}-{}'.format(sent_id,
                                            self.relation_id.split('-')[-1],
                                            self.mention_id.split('-')[-1])
        else:
            relation_id = '{}-{}'.format(self.relation_id.split('-')[-1],
                                         self.mention_id.split('-')[-1])

        return {
            'relation_id': relation_id,
            'relation_type': self.relation_type,
            'relation_subtype': self.relation_subtype,
            'arg1': self.arg1.to_dict(sent_id),
            'arg2': self.arg2.to_dict(sent_id)
        }


@dataclass
class EventArgument:
    entity_id: str
    mention_id: str
    role: str
    text: str

    def to_dict(self, sent_id=None):
        if sent_id:
            entity_id = '{}-{}-{}'.format(sent_id,
                                          self.entity_id.split('-')[-1],
                                          self.mention_id.split('-')[-1])
        else:
            entity_id = '{}-{}'.format(self.entity_id.split('-')[-1],
                                       self.mention_id.split('-')[-1])

        return {
            'entity_id': entity_id,
            'role': self.role,
            'text': recover_escape(self.text),
        }


@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: List[EventArgument]

    def to_dict(self, sent_id=None):
        if sent_id:
            event_id = '{}-{}-{}'.format(sent_id,
                                         self.event_id.split('-')[-1],
                                         self.mention_id.split('-')[-1])
        else:
            event_id = '{}-{}'.format(self.event_id.split('-')[-1],
                                      self.mention_id.split('-')[-1])
        return {
            'event_id': event_id,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            'trigger': self.trigger.to_dict(),
            'arguments': [arg.to_dict(sent_id) for arg in self.arguments]
        }


@dataclass
class Sentence(Span):
    doc_id: str
    sent_id: str
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]

    def to_dict(self):
        return {
            'doc_id': self.doc_id,
            'sent_id': self.sent_id,
            'tokens': [recover_escape(t) for t in self.tokens],
            'entities': [entity.to_dict(self.sent_id) for entity in self.entities],
            'relations': [relation.to_dict(self.sent_id) for relation in self.relations],
            'events': [event.to_dict(self.sent_id) for event in self.events],
            'start': self.start,
            'end': self.end,
            'text': recover_escape(self.text).replace('\t', ' ')
        }


@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]


def sentence_tokenize(sentence, language='english'):
    start, end, text = sentence
    sents = sent_tokenize(text, language='english')

    last = 0
    sents_ = []
    for sent in sents:
        index = text[last:].find(sent)
        if index == -1:
            print(text, sent)
        else:
            sents_.append((last + index + start, last + index + len(sent) + start, sent))
        last += index + len(sent)
    return(sents_)


def read_source_file(path, language='english'):
    data = open(path, 'r', encoding='utf-8').read()

    data = data.replace('\n<a', ' <a').replace('</a>\n', '</a> ')

    min_offset = max(0, data.find('</HEADLINE>'))

    intag = False
    linebreak = True
    sentences = []
    start = end = 0
    sentence = ''
    for i, c in enumerate(data):
        if c == '<':
            intag = True
            linebreak = False
            sentence = ''
        elif c == '>':
            intag = False
            start = end = i + 1
        elif not intag and linebreak:
            if c == '\n':
                if sentence:
                    if start >= min_offset:
                        sentences.append((start, end, sentence))
                    sentence = ''
                start = end = i + 1
            else:
                sentence += c
                end = i + 1
        if c == '\n':
            linebreak = True
            start = end = i + 1
    if sentence:
        if start >= min_offset:
            sentences.append((start, end, sentence))
    for s, e, t in sentences:
        if t != data[s:e]:
            print(t, data[s:e])

    # re-tokenize sentences
    sentences_ = []
    for sent in sentences:
        if not sent[-1].startswith('http') and '</a>' not in sent[-1]:
            sentences_.extend(sentence_tokenize(sent, language=language))
    return sentences_


def read_annotation(path):
    data = open(path, 'r', encoding='utf-8').read()

    soup = BeautifulSoup(data, 'lxml')

    # metadata
    root = soup.find('deft_ere')
    doc_id = root['doc_id']
    source_type = root['source_type']

    # entities
    entity_list = []
    entities_node = root.find('entities')
    if entities_node:
        for entity_node in entities_node.find_all('entity'):
            entity_id = entity_node['id']
            entity_type = entity_node['type']
            for entity_mention_node in entity_node.find_all('entity_mention'):
                mention_id = entity_mention_node['id']
                mention_type = entity_mention_node['noun_type']
                if mention_type == 'NOM':
                    mention_offset = int(entity_mention_node.find('nom_head')['offset'])
                    mention_length = int(entity_mention_node.find('nom_head')['length'])
                    mention_text = entity_mention_node.find('nom_head').text
                else:
                    mention_offset = int(entity_mention_node['offset'])
                    mention_length = int(entity_mention_node['length'])
                    mention_text = entity_mention_node.find('mention_text').text
                entity_list.append(Entity(
                    entity_id=entity_id, mention_id=mention_id,
                    entity_type=entity_type, mention_type=mention_type,
                    start=mention_offset, end=mention_offset + mention_length,
                    text=mention_text
                ))
    fillers_node = root.find('fillers')
    if fillers_node:
        for filler_node in fillers_node.find_all('filler'):
            entity_id = filler_node['id']
            entity_type = filler_node['type']
            if entity_type == 'weapon':
                entity_type = 'WEA'
            elif entity_type == 'vehicle':
                entity_type = 'VEH'
            else:
                continue
            mention_offset = int(filler_node['offset'])
            mention_length = int(filler_node['length'])
            mention_text = filler_node.text
            entity_list.append(
                Entity(
                    entity_id=entity_id, mention_id=entity_id,
                    entity_type=entity_type, mention_type='NOM',
                    start=mention_offset, end=mention_offset + mention_length,
                    text=mention_text
                )
            )


    # relations
    relation_list = []
    relations_node = root.find('relations')
    if relations_node:
        for relation_node in relations_node.find_all('relation'):
            relation_id = relation_node['id']
            relation_type = relation_node['type']
            relation_subtype = relation_node['subtype']
            for relation_mention_node in relation_node.find_all('relation_mention'):
                mention_id = relation_mention_node['id']
                arg1 = relation_mention_node.find('rel_arg1')
                arg2 = relation_mention_node.find('rel_arg2')
                if arg1 and arg2:
                    if arg1.has_attr('entity_id'):
                        arg1_entity_id = arg1['entity_id']
                        arg1_mention_id = arg1['entity_mention_id']
                    else:
                        arg1_entity_id = arg1['filler_id']
                        arg1_mention_id = arg1['filler_id']
                    arg1_role = arg1['role']
                    arg1_text = arg1.text
                    if arg2.has_attr('entity_id'):
                        arg2_entity_id = arg2['entity_id']
                        arg2_mention_id = arg2['entity_mention_id']
                    else:
                        arg2_entity_id = arg2['filler_id']
                        arg2_mention_id = arg2['filler_id']
                    arg2_role = arg2['role']
                    arg2_text = arg2.text
                    relation_list.append(Relation(
                        relation_id=relation_id, mention_id=mention_id,
                        relation_type=relation_type,
                        relation_subtype=relation_subtype,
                        arg1=RelationArgument(entity_id=arg1_entity_id,
                                              mention_id=arg1_mention_id,
                                              role=arg1_role,
                                              text=arg1_text),
                        arg2=RelationArgument(entity_id=arg2_entity_id,
                                              mention_id=arg2_mention_id,
                                              role=arg2_role,
                                              text=arg2_text)))

    # events
    event_list = []
    events_node = root.find('hoppers')
    if events_node:
        for event_node in events_node.find_all('hopper'):
            event_id = event_node['id']
            for event_mention_node in event_node.find_all('event_mention'):
                trigger = event_mention_node.find('trigger')
                trigger_offset = int(trigger['offset'])
                trigger_length = int(trigger['length'])
                arguments = []
                for arg in event_mention_node.find_all('em_arg'):
                    if arg['realis'] == 'false':
                        continue
                    if arg.has_attr('entity_id'):
                        arguments.append(EventArgument(
                            entity_id=arg['entity_id'],
                            mention_id=arg['entity_mention_id'],
                            role=arg['role'],
                            text=arg.text))
                    elif arg.has_attr('filler_id'):
                        arguments.append(EventArgument(
                            entity_id=arg['filler_id'],
                            mention_id=arg['filler_id'],
                            role=arg['role'],
                            text=arg.text
                        ))
                event_list.append(Event(
                    event_id=event_id,
                    mention_id=event_mention_node['id'],
                    event_type=event_mention_node['type'],
                    event_subtype=event_mention_node['subtype'],
                    trigger=Span(start=trigger_offset,
                                         end=trigger_offset + trigger_length,
                                         text=trigger.text),
                    arguments=arguments))
    return doc_id, source_type, entity_list, relation_list, event_list


def clean_entities(entities, sentences):
    sentence_entities = [[] for _ in range(len(sentences))]
    for entity in entities:
        start, end = entity.start, entity.end
        for i, (s, e, _) in enumerate(sentences):
            if start >= s and end <= e:
                entity.remove_space()
                sentence_entities[i].append(entity)
                break
    # remove overlapping entities
    sentence_entities_ = [[] for _ in range(len(sentences))]
    for i, entities in enumerate(sentence_entities):
        if not entities:
            continue
        entities.sort(key=lambda x: (x.end - x.start), reverse=True)
        chars = [0] * max([x.end for x in entities])
        for entity in entities:
            overlap = False
            for j in range(entity.start, entity.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if overlap:
                continue
            else:
                chars[entity.start:entity.end] = [1] * (entity.end - entity.start)
                sentence_entities_[i].append(entity)
        sentence_entities_[i].sort(key=lambda x: x.start)

    return sentence_entities_


def clean_events(events, sentence_entities, sentences):
    """
    :param events (list): A list of events.
    :param sentence_entities (list): A cleaned list of entities.
    :param sentences (list): A list of sentences.
    """
    sentence_events = [[] for _ in range(len(sentences))]
    for event in events:
        start, end = event.trigger.start, event.trigger.end
        for i, (s, e, _) in enumerate(sentences):
            if start >= s and end <= e:
                event.trigger.remove_space()
                # clean the argument list
                arguments = []
                entities = sentence_entities[i]
                for argument in event.arguments:
                    entity_id = argument.entity_id
                    mention_id = argument.mention_id
                    for entity in entities:
                        if (entity.entity_id == entity_id and
                                entity.mention_id == mention_id):
                            arguments.append(argument)
                event_ = Event(
                    event_id=event.event_id, mention_id=event.mention_id,
                    event_type=event.event_type, event_subtype=event.event_subtype,
                    trigger=event.trigger, arguments=arguments
                )
                sentence_events[i].append(event_)

    # remove overlapping events
    sentence_events_ = [[] for _ in range(len(sentences))]
    for i, events in enumerate(sentence_events):
        if not events:
            continue
        events.sort(key=lambda x: (x.trigger.end - x.trigger.start),
                    reverse=True)
        chars = [0] * max([x.trigger.end for x in events])
        for event in events:
            overlap = False
            for j in range(event.trigger.start, event.trigger.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if overlap:
                continue
            else:
                chars[event.trigger.start:event.trigger.end] = \
                    [1] * (event.trigger.end - event.trigger.start)
                sentence_events_[i].append(event)
        sentence_events_[i].sort(key=lambda x: x.trigger.start)
    return sentence_events_


def clean_relations(relations, sentence_entities, sentences):
    sentence_relations = [[] for _ in range(len(sentences))]
    for relation in relations:
        entity_id_1, mention_id_1 = relation.arg1.entity_id, relation.arg1.mention_id
        entity_id_2, mention_id_2 = relation.arg2.entity_id, relation.arg2.mention_id
        for i, entities in enumerate(sentence_entities):
            arg1_in_sent = arg2_in_sent = False
            for entity in entities:
                if entity.entity_id == entity_id_1 and entity.mention_id == mention_id_1:
                    arg1_in_sent = True
                if entity.entity_id == entity_id_2 and entity.mention_id == mention_id_2:
                    arg2_in_sent = True
            if arg1_in_sent and arg2_in_sent:
                sentence_relations[i].append(relation)
                break
            elif arg1_in_sent != arg2_in_sent:
                # stop searching because we find only one entity in the current
                # sentence.
                break
    return sentence_relations


def tokenize(sentence, entities, events, language='english'):
    start, end, text = sentence
    text = mask_escape(text)
    # split the sentence into chunks
    splits = {0, len(text)}
    # print(text)
    for entity in entities:
        splits.add(entity.start - start)
        splits.add(entity.end - start)
    for event in events:
        splits.add(event.trigger.start - start)
        splits.add(event.trigger.end - start)
    splits = sorted(list(splits))
    chunks = [(splits[i], splits[i + 1], text[splits[i]:splits[i + 1]])
              for i in range(len(splits) - 1)]

    # tokenize each chunk
    chunks = [(s, e, t, wordpunct_tokenize(t, language=language)) for s, e, t in chunks]


    # merge chunks and add word offsets
    tokens = []
    for chunk_start, chunk_end, chunk_text, chunk_tokens in chunks:
        last = 0
        chunk_tokens_ = []
        for token in chunk_tokens:
            token_start = chunk_text[last:].find(token)
            if token_start == -1:
                raise ValueError('Cannot find token {} in {}'.format(token, text))
            token_end = token_start + len(token)
            chunk_tokens_.append((token_start + start + last + chunk_start,
                                  token_end + start + last + chunk_start,
                                  unmask_escape(token)))
            last += token_end
        # print(chunk_tokens, chunk_tokens_)
        tokens.extend(chunk_tokens_)
    return tokens


def extract(source_path, ere_path, language='english'):
    sentences = read_source_file(source_path, language=language)
    doc_id, source_type, entities, relations, events = read_annotation(ere_path)

    # remove entities and events out of extracted sentences
    sentence_entities = clean_entities(entities, sentences)
    sentence_events = clean_events(events, sentence_entities, sentences)
    sentence_relations = clean_relations(relations, sentence_entities, sentences)

    # tokenization
    sentence_tokens = [tokenize(sent, ent, evt, language=language) for sent, ent, evt
              in zip(sentences, sentence_entities, sentence_events)]

    # convert span character offsets to token index offsets
    sentence_objs = []
    for i, (tokens, entities, events, relations, sentence) in enumerate(zip(
            sentence_tokens, sentence_entities, sentence_events,
            sentence_relations, sentences)):
        for entity in entities:
            entity.char_offsets_to_token_offsets(tokens)
        for event in events:
            event.trigger.char_offsets_to_token_offsets(tokens)
        sent_id = '{}-{}'.format(doc_id, i)
        sentence_objs.append(Sentence(
            doc_id=doc_id, sent_id=sent_id,
            tokens=[t for _, _, t in tokens],
            entities=entities,
            relations=relations,
            events=events,
            start=sentence[0],
            end=sentence[1],
            text=sentence[-1]
        ))
    return Document(doc_id=doc_id, sentences=sentence_objs)


def process_batch(input_dir, output_file, dataset='normal', language='english'):
    if dataset == 'normal':
        source_files = glob.glob(os.path.join(input_dir, 'source', 'cmptxt', '*.txt'))
    elif dataset == 'r2v2':
        source_files = glob.glob(os.path.join(input_dir, 'source', '*.txt'))
    elif dataset == 'parallel':
        source_files = glob.glob(os.path.join(input_dir, 'eng', 'translation', '*.txt'))
    elif dataset == 'spanish':
        source_files = glob.glob(os.path.join(input_dir, 'source', '**', '*.txt'))
    with open(output_file, 'w', encoding='utf-8') as w:
        for source_file in source_files:
            doc_id = os.path.basename(source_file).replace('.txt', '') \
                .replace('.cmp', '').replace('.mp', '')
            # print(doc_id)
            if dataset == 'normal':
                annotation_file = os.path.join(input_dir, 'ere', 'cmptxt', '{}.rich_ere.xml'.format(doc_id))
            elif dataset == 'r2v2':
                annotation_file = os.path.join(input_dir, 'ere', '{}.rich_ere.xml'.format(doc_id))
            elif dataset == 'parallel':
                annotation_file = os.path.join(input_dir, 'eng', 'ere', '{}.rich_ere.xml'.format(doc_id))
            elif dataset == 'spanish':
                annotation_file = source_file.replace('/source/', '/ere/').replace('.txt', '.rich_ere.xml')
            doc = extract(source_file, annotation_file, language=language)
            for sent in doc.sentences:
                w.write(json.dumps(sent.to_dict()) + '\n')


def ere_to_oneie(input_file, output_file, tokenizer=None):
    skip_num = 0
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(output_file, 'w', encoding='utf-8') as w:
        for line in r:
            inst = json.loads(line)
            # tokens
            tokens = inst['tokens']
            pieces = [tokenizer.tokenize(t) for t in tokens]
            token_lens = [len(x) for x in pieces]
            if 0 in token_lens:
                skip_num += 1
                continue
            pieces = [p for ps in pieces for p in ps]
            sentence = inst['text']
            # entities
            entity_mentions = []
            entity_text = {}
            for entity in inst['entities']:
                entity_mentions.append({
                    'id': entity['entity_id'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'entity_type': entity['entity_type'],
                    'mention_type': entity['mention_type'],
                    'text': entity['text']
                })
                entity_text[entity['entity_id']] = entity['text']
            # relations
            relation_mentions = []
            for relation in inst['relations']:
                relation_mentions.append({
                    'id': relation['relation_id'],
                    'relation_type': relation_type_mapping[relation['relation_type']],
                    'arguments': [
                        {
                            'entity_id': relation['arg1']['entity_id'],
                            'role': 'Arg-1',
                            'text': entity_text[relation['arg1']['entity_id']]
                        },
                        {
                            'entity_id': relation['arg2']['entity_id'],
                            'role': 'Arg-2',
                            'text': entity_text[relation['arg2']['entity_id']]
                        },
                    ]
                })
            # events
            event_mentions = []
            for event in inst['events']:
                event_mentions.append({
                    'id': event['event_id'],
                    'event_type': event_type_mapping['{}:{}'.format(
                        event['event_type'], event['event_subtype'])],
                    'trigger': {
                        'start': event['trigger']['start'],
                        'end': event['trigger']['end'],
                        'text': event['trigger']['text']},
                    'arguments': [{
                        'entity_id': arg['entity_id'],
                        'text': entity_text[arg['entity_id']],
                        'role': role_type_mapping[arg['role']]
                    } for arg in event['arguments']]
                })

            w.write(json.dumps({
                'doc_id': inst['doc_id'],
                'sent_id': inst['sent_id'],
                'tokens': tokens,
                'pieces': pieces,
                'token_lens': token_lens,
                'entity_mentions': entity_mentions,
                'relation_mentions': relation_mentions,
                'event_mentions': event_mentions
            }) + '\n')
    print('#Skip: {}'.format(skip_num))