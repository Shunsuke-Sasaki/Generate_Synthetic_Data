# -*- coding: utf-8 -*-
from collections import Counter
from typing import List, Dict, Optional

import os
import json
import numpy as np

from tasks.task_abc import Question, Document, Task
from utils.io_utils import jload_list, jload
from utils.prompt_utils import (
    format_name, uncapitalize_first, second_last_character,
    OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES,
    OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS,
    OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS,
    QUALITY_FEW_SHOT_COT_PROMPT,
)

# =========================
# 日本語プロンプト（自前データ用）
# =========================
JA_SYSTEM_GENERATE_ENTITIES = (
    "あなたは長文から主要な固有表現・専門用語を抽出する抽出器です。"
    "入力文書から重複のないエンティティを10〜30個ほど抽出し、短い要約も返してください。"
    "必ず JSON 形式 {\"entities\": [\"...\"], \"summary\": \"...\"} で返答し、"
    "本文にない情報は作らず、日本語で簡潔に記述してください。"
)

JA_SYSTEM_TWO_ENTITY_REL = (
    "以下の文書内容と与えられた2つのエンティティに基づき、両者の関係を日本語で1〜2段落で説明してください。"
    "本文に即し、推測や一般論は避けてください。"
)

JA_SYSTEM_THREE_ENTITY_REL = (
    "以下の文書内容と与えられた3つのエンティティに基づき、三者の関係（相互作用・因果・包含・対比など）を"
    "日本語で1〜2段落で具体的に説明してください。本文にない内容は避けてください。"
)


# =========================
# 既存 QuALITY 用クラス群（そのまま）
# =========================
class QuALITYQuestion(Question):
    def __init__(self,
                 statement: str,
                 options: List[str],
                 answer: str,
                 ishard: bool,
                 attempts: List[Dict] = [dict()],
                 **kwargs):
        statement_dict = dict(content=statement, options=options)
        super().__init__(statement_dict, answer, attempts)
        self.ishard = ishard

    def _formatted_choice(self):
        formatted = ""
        for i, option in enumerate(self.statement['options']):
            letter = chr(65 + i)  # A,B,C,D...
            formatted += f"{letter}. {option}\n"
        return formatted

    def prompt(
        self,
        document_context: Optional[str],
        add_thought_process: bool,
        sep_after_question: str
    ):
        formatted = "### Question\n"

        if document_context is None:
            formatted += f"{self.statement['content']} There is only one correct choice.{sep_after_question}"
        else:
            formatted += f"{document_context} {uncapitalize_first(self.statement['content'])} There is only one correct choice.{sep_after_question}"

        formatted += "### Choices\n"
        formatted += self._formatted_choice()

        if add_thought_process:
            if sep_after_question == '\n\n':
                formatted += "\n"
            formatted += "### Thought Process and Answer\n"
            formatted += "Thought process:"

        return formatted

    def llama_parse_answer(self, raw_output: str):
        if raw_output is None:
            return dict()
        else:
            answer_index = second_last_character(raw_output)
            if answer_index is not None:
                answer_content = self.statement['options'][answer_index]
            else:
                answer_content = None
            return dict(reasoning=raw_output,
                        answer_index=answer_index,
                        answer_content=answer_content)

    def _iscorrect(self, attempt: Dict):
        return self.answer == chr(attempt['answer_index'] + 65)

    def iscorrect(self, attempt_index: int = 0):
        return self._iscorrect(self.attempts[attempt_index])

    def asdict(self):
        return dict(
            statement=self.statement['content'],
            options=self.statement['options'],
            answer=self.answer,
            ishard=self.ishard,
            attempts=self.attempts,
            formatted_prompt=self.formatted_prompt
        )

    def majority_vote(self, n_samples: int):
        if len(self.attempts) > 0:
            self.attempts = self.attempts[:n_samples]
            indices = [attempt['answer_index'] for attempt in self.attempts]
            counts = Counter(indices)
            most_freq = max(counts, key=counts.get)
            for attempt in self.attempts:
                if attempt['answer_index'] == most_freq:
                    self.attempts = [attempt]
        else:
            self.attempts = [dict()]


class QuALITYArticle(Document):
    def __init__(self, text: str, questions: List[Dict],
                 title: str, author: str, year: str, topic: str, **kwargs):
        questions = [QuALITYQuestion(**qdict) for qdict in questions]
        super().__init__(text, questions)
        self.title = title
        self.author = author
        self.year = year
        self.topic = topic

    @property
    def uid(self):
        return ' by '.join([self.title, self.author])

    @property
    def content(self):
        result = f"\"{self.title}\", {format_name(self.author)}, {self.year}."
        result += f"\n {self.text}"
        return result

    @property
    def _article_context(self):
        return f"In the context of \"{self.title}\", written by {format_name(self.author)} in {self.year},"

    def question_prompts(self, add_document_context: bool, add_thought_process: bool, sep_after_question: str):
        prompts = []
        for q in self.questions:
            prompts.append(
                q.prompt(
                    self._article_context if add_document_context else None,
                    add_thought_process,
                    sep_after_question)
            )
        return prompts

    def asdict(self):
        return dict(title=self.title,
                    author=self.author,
                    year=self.year,
                    topic=self.topic,
                    text=self.text,
                    questions=[q.asdict() for q in self.questions])


# =========================
# 自前データ用の極小ドキュメント（EntiGraphは質問を使わないため簡素でOK）
# =========================
class GenericArticle(Document):
    """自前JSONL（uid/raw）1行=1文書を包む軽量ドキュメント"""
    def __init__(self, uid: str, text: str):
        super().__init__(text, questions=[])  # EntiGraph 側では questions を使わない
        self._uid = uid

    @property
    def uid(self):
        return self._uid

    @property
    def content(self):
        # そのまま本文を返す（タイトル・著者フォーマットは不要）
        return self.text


class QuALITY(Task):
    """
    既存:
        QuALITY('train'), QuALITY('all'), QuALITY('test')
    追加:
        環境変数 QUALITY_USE_CUSTOM=1 がセットされている場合、
        SOURCE_JSONL（既定: data/dataset/source/mycorpus.jsonl）から {uid, raw} を読み込む。
        プロンプトは日本語版（JA_*）に切替。
    """
    # 既存デフォルト（英語）をクラス変数に残す
    openai_system_generate_entities = OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES
    openai_system_generate_two_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS
    openai_system_generate_three_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS
    llama_cot_prompt = QUALITY_FEW_SHOT_COT_PROMPT


    @staticmethod
    def _load_custom_jsonl(path: str):
        """各行が {uid, raw} の JSONL を読み込む"""
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                uid = obj.get("uid")
                raw = obj.get("raw")
                if uid and raw:
                    docs.append(dict(uid=uid, raw=raw))
        return docs

    def _create_documents(self):
        """QuALITY（英語）標準データ→Article+Questions"""
        documents = []
        for adict in self._data:
            questions = []
            for qdict in adict['questions']:
                question = dict(statement=qdict['question'],
                                options=qdict['options'],
                                answer=chr(int(qdict['gold_label']) - 1 + 65),
                                ishard=bool(qdict['difficult']))
                questions.append(question)
            questions = sorted(questions, key=lambda x: x['statement'])
            document = QuALITYArticle(
                title=adict['title'],
                author=adict['author'],
                text=adict['article'],
                year=adict['year'],
                topic=adict['topic'],
                questions=questions
            )
            documents.append(document)
        super().__init__('quality', sorted(documents, key=lambda x: x.title))

    def _create_documents_from_custom(self, custom_records: List[Dict]):
        """自前JSONL（uid/raw）→ GenericArticle"""
        documents = []
        for rec in custom_records:
            documents.append(GenericArticle(uid=rec['uid'], text=rec['raw']))
        # uid で安定ソート
        documents = sorted(documents, key=lambda x: x.uid)
        super().__init__('quality', documents)

    def _dedup(self):
        deuped_documents = {}
        for document in self.documents:
            key = document.uid
            if key not in deuped_documents:
                deuped_documents[key] = document
            else:
                # QuALITYArticle のときのみ質問を併合（GenericArticle は questions が空）
                deuped_documents[key].questions += getattr(document, "questions", [])
        self.documents = list(deuped_documents.values())

    def __init__(self, split: str):
        self.split = split

        # --- 自前データモードの判定（環境変数で切り替え） ---
        use_custom = os.environ.get("QUALITY_USE_CUSTOM", "0") in ("1", "true", "True")
        source_jsonl = os.environ.get("SOURCE_JSONL", "data/dataset/source/mycorpus.jsonl")

        if use_custom:
            # 自前 JSONL を読み込み、GenericArticle と日本語プロンプトをセット
            custom = QuALITY._load_custom_jsonl(source_jsonl)
            if not custom:
                raise RuntimeError(f"[QuALITY] No valid records in {source_jsonl} (need uid/raw).")
            self._create_documents_from_custom(custom)

            # —— 日本語プロンプトに切替（インスタンス属性で上書き）——
            self.openai_system_generate_entities = JA_SYSTEM_GENERATE_ENTITIES
            self.openai_system_generate_two_entity_relations = JA_SYSTEM_TWO_ENTITY_REL
            self.openai_system_generate_three_entity_relations = JA_SYSTEM_THREE_ENTITY_REL
            self.llama_cot_prompt = ""  # 自前生成では使わないため空でも可
            return

        # --- 既存の QuALITY データフロー（英語） ---
        if split in ['train']:
            self._data = QuALITY._load_split(split)
            self._create_article()  # NOTE: 既存コードと互換（おそらく _create_article は Task 側ユーティリティ）
        elif split in ['all', '50']:
            self._data = QuALITY._load_split('train')  # + QuALITY._load_split('dev')
            self._create_documents()
            self._dedup()
            if split == '50':
                self.documents = self.documents[:50]
        elif split == 'test':
            super().__init__('quality', None)
        else:
            raise ValueError(f"Invalid split: {split}")

    # 以降は元のまま
    def load_attempts_json(self, file_path: str):
        loaded_articles_data = jload(file_path)
        attempted_articles = []
        for adict in loaded_articles_data:
            article = QuALITYArticle(**adict)
            attempted_articles.append(article)
        super().__init__('quality', sorted(attempted_articles, key=lambda x: x.title))

    def all_questions(self, add_document_context: bool, add_thought_process: bool, sep_after_question: str):
        prompts = []
        for document in self.documents:
            prompts += document.question_prompts(add_document_context, add_thought_process, sep_after_question)
        return prompts

    @staticmethod
    def _attempts_stats(attempt_index: int, documents: List[QuALITYArticle]):
        attempted_hard_q = 0
        attempted_non_hard_q = 0
        correct_hard_q = 0
        correct_non_hard_q = 0

        for article in documents:
            for question in article.questions:
                if question.attempts[attempt_index]:
                    try:
                        if question.attempts[attempt_index]['answer_index'] in [0, 1, 2, 3]:
                            if question.ishard:
                                attempted_hard_q += 1
                                if question.iscorrect(attempt_index):
                                    correct_hard_q += 1
                            else:
                                attempted_non_hard_q += 1
                                if question.iscorrect(attempt_index):
                                    correct_non_hard_q += 1
                    except KeyError:
                        print(f"KeyError: {question.attempts[attempt_index]}")

        return dict(attempted_hard_q=attempted_hard_q,
                    attempted_non_hard_q=attempted_non_hard_q,
                    correct_hard_q=correct_hard_q,
                    correct_non_hard_q=correct_non_hard_q)

    @staticmethod
    def _question_stats(documents: List[QuALITYArticle]):
        answerhist = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for article in documents:
            for question in article.questions:
                answerhist[question.answer] += 1

        hard_q = 0
        non_hard_q = 0
        for article in documents:
            for question in article.questions:
                if question.ishard:
                    hard_q += 1
                else:
                    non_hard_q += 1
        return dict(answerhist=answerhist, hard_q=hard_q, non_hard_q=non_hard_q)

    @staticmethod
    def _performance_stats_for_documents(documents: List[QuALITYArticle]):
        question_stats = QuALITY._question_stats(documents)

        def calculate_one_attempt(attempts_stats: Dict):
            def _div_nan_if_zero(a, b):
                if a == 0 or b == 0:
                    return np.nan
                return a / b

            hard_attempt_rate = _div_nan_if_zero(attempts_stats['attempted_hard_q'], question_stats['hard_q'])
            hard_accuracy = _div_nan_if_zero(attempts_stats['correct_hard_q'], attempts_stats['attempted_hard_q'])
            non_hard_attempt_rate = _div_nan_if_zero(attempts_stats['attempted_non_hard_q'], question_stats['non_hard_q'])
            non_hard_accuracy = _div_nan_if_zero(attempts_stats['correct_non_hard_q'], attempts_stats['attempted_non_hard_q'])
            overall_attempt_rate = _div_nan_if_zero(
                attempts_stats['attempted_hard_q'] + attempts_stats['attempted_non_hard_q'],
                question_stats['hard_q'] + question_stats['non_hard_q'])
            overall_accuracy = _div_nan_if_zero(
                attempts_stats['correct_hard_q'] + attempts_stats['correct_non_hard_q'],
                attempts_stats['attempted_hard_q'] + attempts_stats['attempted_non_hard_q'])

            return dict(
                hard_attempt_rate=hard_attempt_rate,
                hard_accuracy=hard_accuracy,
                non_hard_attempt_rate=non_hard_attempt_rate,
                non_hard_accuracy=non_hard_accuracy,
                overall_attempt_rate=overall_attempt_rate,
                overall_accuracy=overall_accuracy
            )

        attempts_stats = QuALITY._attempts_stats(0, documents)
        one_attempt = calculate_one_attempt(attempts_stats)

        result = dict()
        for key, value in one_attempt.items():
            result[key] = dict(mean=value, std=0)
        return result

    def performance_stats(self):
        total_documents = len(self.documents)
        val_documents = self.documents[:int(0.2 * total_documents)]
        test_documents = self.documents[int(0.2 * total_documents):]
        result = dict(val=QuALITY._performance_stats_for_documents(val_documents),
                      test=QuALITY._performance_stats_for_documents(test_documents))
        return result
