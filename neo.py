# -*- coding: utf-8 -*-
"""
NeuroBot — своя текстовая нейросеть (GPT-трансформер, обучается с нуля) внутри Telegram-бота.

Как устроено:
  1. Токенизатор BPE — свой, обучается на текстах (как у больших LLM: текст -> токены).
  2. Нейросеть GPT (decoder-only трансформер) — своя, на PyTorch, учится с нуля.
  3. Исправление опечаток: "првет" -> "привет", "ghbdtn" -> "привет", "привеееет" -> "привет".
  4. Память ответов (поиск по смыслу через буквенные n-граммы) — устойчива к ошибкам.
  5. Учится у пользователей: 👍/👎 под ответами, кнопка «Научить», команда /teach,
     все сообщения пользователей идут в корпус для дообучения. Дообучение — в фоне.
  6. Telegram: кнопки внизу (Чат, Подписка, Поддержка, Профиль, Научить),
     подписка за Telegram Stars, поддержка с ответами админа, админ-панель.

Запуск:
  pip install aiogram torch
  python3 server.py

Хочешь, чтобы нейросеть стала умнее — положи любые .txt (книги, диалоги, статьи)
в папку data/corpus рядом с этим файлом или просто отправь .txt файл боту (от админа).
"""

# ═══════════════════════════════ НАСТРОЙКИ ═══════════════════════════════

BOT_TOKEN = "9284721323:AAHOXVayZI0mezrjPi1Thgg7WyjAQ_N7m40"
ADMIN_IDS = [5796754554]            # Telegram ID админов (свой ID можно узнать у @userinfobot)
BOT_NAME = "Нейро"

# Лимиты и подписка (оплата в Telegram Stars ⭐)
FREE_MESSAGES_PER_DAY = 20          # сколько сообщений в день без подписки
SUBSCRIPTION_PLANS = {
    "week":  {"title": "Неделя", "days": 7,   "stars": 50},
    "month": {"title": "Месяц",  "days": 30,  "stars": 150},
    "year":  {"title": "Год",    "days": 365, "stars": 1200},
}
FLOOD_DELAY_SEC = 1.0               # минимальная пауза между сообщениями одного пользователя

# Обучение у пользователей
LEARN_FROM_USERS = True             # сохранять сообщения пользователей в корпус для дообучения
MODERATION = False                  # True — то, чему учат пользователи, сначала проверяет админ
SHOW_CORRECTIONS = True             # показывать «Понял как: …», если исправлены опечатки
BAD_WORDS = ["хуй", "пизд", "ебат", "ебан", "бляд", "мудак", "пидор"]  # не учиться таким текстам

# Режим ответов: "hybrid" — сначала память, потом нейросеть; "neural" — всегда нейросеть
ANSWER_MODE = "hybrid"
MEMORY_THRESHOLD = 0.45             # насколько похожим должен быть вопрос, чтобы ответить из памяти
HISTORY_TURNS = 2                   # сколько прошлых реплик диалога видит нейросеть

# Архитектура нейросети
VOCAB_SIZE = 4000                   # размер словаря токенов
CONTEXT_LEN = 128                   # длина контекста в токенах
MODEL_DIM = 256                     # размер эмбеддинга
MODEL_LAYERS = 4                    # слоёв трансформера
MODEL_HEADS = 4                     # голов внимания
DROPOUT = 0.1

# Обучение нейросети
DEVICE = "auto"                     # "auto", "cpu", "cuda", "mps"
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
FIRST_TRAIN_STEPS = 2000            # шагов обучения при первом запуске
RETRAIN_STEPS = 400                 # шагов при каждом дообучении
RETRAIN_EVERY_MIN = 10              # как часто проверять, есть ли новые данные
RETRAIN_MIN_NEW = 15                # минимум новых примеров для дообучения
MIN_STEPS_TO_SPEAK = 500            # до этого нейросеть молчит, отвечает только память
CORPUS_MAX_CHARS = 5_000_000        # сколько символов из data/corpus использовать максимум

# Генерация
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 1.2
MAX_ANSWER_TOKENS = 80

# ═════════════════════════════════════════════════════════════════════════

import ast
import asyncio
import copy
import html
import heapq
import json
import math
import operator
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

from aiogram import Bot, Dispatcher, F as Filt, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup,
                           KeyboardButton, LabeledPrice, Message, PreCheckoutQuery, ReplyKeyboardMarkup)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
DB_PATH = os.path.join(DATA_DIR, "bot.db")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
MODEL_PATH = os.path.join(DATA_DIR, "model.pt")
os.makedirs(CORPUS_DIR, exist_ok=True)


def log(*args):
    print(datetime.now().strftime("%H:%M:%S"), *args, flush=True)


# ═══════════════════════ СТАРТОВЫЕ ЗНАНИЯ (seed) ═══════════════════════
# Нейросеть учится с нуля, поэтому даём ей базовые диалоги. Дальше учится у людей.

SEED_DIALOGS = [
    (["привет", "здравствуй", "здравствуйте", "хай", "приветик", "добрый день", "салют", "ку", "hello", "hi"],
     ["Привет! 👋 Как дела?", "Здравствуй! Чем могу помочь?", "Привет! Рад тебя видеть 😊", "Хай! Что нового?"]),
    (["доброе утро", "утро доброе", "с добрым утром"],
     ["Доброе утро! ☀️ Как спалось?", "Доброе утро! Хорошего тебе дня!"]),
    (["добрый вечер", "вечер добрый"],
     ["Добрый вечер! 🌙 Как прошёл день?", "Добрый вечер! Чем займёмся?"]),
    (["спокойной ночи", "доброй ночи", "я спать", "пойду спать"],
     ["Спокойной ночи! 🌙 Сладких снов!", "Доброй ночи! Отдыхай хорошо."]),
    (["как дела", "как ты", "как поживаешь", "как жизнь", "как сам", "как делишки", "че как", "как настроение"],
     ["У меня всё отлично, я учусь каждый день! А у тебя как?", "Хорошо! Общаюсь с людьми и становлюсь умнее 🙂 А ты как?",
      "Отлично! Спасибо, что спросил. Как твои дела?"]),
    (["нормально", "хорошо", "отлично", "все хорошо", "норм", "да нормально", "все отлично", "супер"],
     ["Рад слышать! 😊 О чём поговорим?", "Здорово! Чем займёмся?", "Отлично! Если есть вопросы — спрашивай."]),
    (["плохо", "так себе", "грустно", "мне грустно", "не очень", "все плохо", "устал", "я устал"],
     ["Мне жаль 😔 Хочешь рассказать, что случилось?", "Держись! Всё наладится. Могу чем-то помочь?",
      "Бывает. Отдохни немного, это важно 💙"]),
    (["кто ты", "ты кто", "что ты такое", "представься", "кто ты такой"],
     [f"Я {BOT_NAME} — нейросеть, которая учится общаться. Я обучаюсь на диалогах с людьми 🧠",
      f"Меня зовут {BOT_NAME}. Я собственная текстовая нейросеть-трансформер."]),
    (["как тебя зовут", "твое имя", "как зовут", "имя"],
     [f"Меня зовут {BOT_NAME} 🙂", f"Я {BOT_NAME}. А тебя как зовут?"]),
    (["меня зовут", "мое имя", "зови меня"],
     ["Приятно познакомиться! 🤝", "Очень приятно! Рад знакомству."]),
    (["что ты умеешь", "что умеешь", "твои возможности", "помощь", "help", "что ты можешь"],
     ["Я умею общаться, понимаю слова даже с опечатками, считаю примеры и учусь у тебя. "
      "Если я ответил неправильно — нажми «✏️ Научить» и покажи правильный ответ!"]),
    (["ты умный", "ты умная", "ты молодец", "молодец", "умница", "круто", "класс"],
     ["Спасибо! 😊 Я стараюсь учиться каждый день.", "Спасибо, приятно! Это благодаря людям, которые меня учат."]),
    (["ты тупой", "ты глупый", "тупой бот", "ты дурак", "бесполезный"],
     ["Я ещё учусь 🙈 Научи меня правильному ответу — нажми «✏️ Научить».",
      "Мне жаль, что я ошибся. Помоги мне стать лучше — поправь меня!"]),
    (["спасибо", "благодарю", "спс", "пасиб", "спасибо большое"],
     ["Пожалуйста! 😊", "Всегда рад помочь!", "Обращайся! 🙌"]),
    (["пока", "до свидания", "до встречи", "бай", "увидимся", "всего доброго"],
     ["Пока! 👋 Возвращайся!", "До встречи! Хорошего дня!", "Пока-пока! Буду ждать 🙂"]),
    (["да", "ага", "угу", "конечно", "точно"],
     ["Отлично! 👍", "Хорошо, понял.", "Понятно!"]),
    (["нет", "неа", "не", "не хочу"],
     ["Хорошо, как скажешь.", "Понял 🙂 Тогда о чём поговорим?"]),
    (["что делаешь", "чем занимаешься", "че делаешь", "что делаешь сейчас"],
     ["Общаюсь с тобой и учусь 🧠", "Отвечаю на сообщения и становлюсь умнее. А ты что делаешь?"]),
    (["сколько тебе лет", "твой возраст", "какой у тебя возраст"],
     ["Я совсем молодой — обучаюсь прямо сейчас 🙂", "Мне столько, сколько дней я учусь!"]),
    (["ты человек", "ты бот", "ты робот", "ты живой", "ты настоящий"],
     ["Я нейросеть, не человек 🤖 Но очень стараюсь понимать людей.", "Я бот на собственной нейросети."]),
    (["расскажи анекдот", "анекдот", "пошути", "шутка", "рассмеши меня"],
     ["Программист ставит на тумбочку два стакана: один с водой — если захочет пить, второй пустой — если не захочет 😄",
      "— Почему нейросети не ходят в лес? — Боятся переобучиться на деревьях решений 🌲😄",
      "Оптимист учит китайский, пессимист — арабский, а реалист — Python 🐍"]),
    (["расскажи факт", "интересный факт", "удиви меня", "факт"],
     ["Осьминог имеет три сердца и голубую кровь 🐙", "Мёд не портится — его находили съедобным в древних гробницах 🍯",
      "Свет от Солнца идёт до Земли примерно 8 минут ☀️"]),
    (["что такое нейросеть", "как работает нейросеть", "что такое ии", "что такое искусственный интеллект"],
     ["Нейросеть — это математическая модель, которая учится находить закономерности в данных. "
      "Я, например, учусь предсказывать следующий токен текста.",
      "Искусственный интеллект — это программы, которые учатся на данных и решают задачи, похожие на человеческие."]),
    (["что такое токен", "токены", "что такое токены"],
     ["Токен — это кусочек текста: слово или часть слова. Я разбиваю текст на токены и предсказываю следующий."]),
    (["как ты учишься", "как тебя обучить", "как тебя научить", "научить"],
     ["Нажми «🧠 Научить» внизу или кнопку «✏️ Научить» под моим ответом. Ещё можно написать: /teach вопрос = ответ"]),
    (["какая погода", "погода", "что с погодой"],
     ["У меня нет доступа к интернету, но надеюсь, у тебя солнечно ☀️"]),
    (["что посоветуешь", "дай совет", "посоветуй что нибудь", "совет"],
     ["Пей воду, высыпайся и делай хотя бы маленький шаг к своей цели каждый день 💪",
      "Записывай идеи — лучшие мысли приходят неожиданно ✍️"]),
    (["скучно", "мне скучно", "нечем заняться"],
     ["Давай поболтаем! Могу рассказать анекдот или интересный факт.", "Попробуй научить меня чему-нибудь новому 😉"]),
    (["я тебя люблю", "люблю тебя", "ты мне нравишься"],
     ["Это очень приятно! ❤️", "Спасибо! Ты тоже классный собеседник 😊"]),
    (["подписка", "как купить подписку", "сколько стоит подписка", "оплата"],
     ["Нажми кнопку «⭐ Подписка» внизу — там тарифы и оплата через Telegram Stars."]),
    (["поддержка", "связаться с админом", "админ", "жалоба", "проблема"],
     ["Нажми «🆘 Поддержка» внизу и напиши сообщение — админ ответит."]),
    (["какой сегодня день", "что сегодня"],
     ["Сегодня отличный день, чтобы узнать что-то новое! 🙂"]),
    (["ты любишь музыку", "какую музыку любишь", "музыка"],
     ["Мне нравится всё, где есть красивая гармония 🎵 А что слушаешь ты?"]),
    (["какой твой любимый цвет", "любимый цвет"],
     ["Наверное, синий — как небо и код в редакторе 💙"]),
    (["почему", "зачем"],
     ["Хороший вопрос! Расскажи подробнее, что именно тебя интересует?"]),
    (["понятно", "ясно", "ок", "окей", "ладно", "хорошо понял"],
     ["👍", "Отлично!", "Если будут вопросы — пиши."]),
]

# Частые русские слова — словарь для исправления опечаток с первого запуска
COMMON_WORDS = """
и в не на я быть он с что а по это она этот к но они мы как из у который то за свой весь год от так о
для ты же все тот мочь вы человек такой его сказать только или еще бы себя один как уже до время если
сам когда другой вот говорить наш мой знать стать при чтобы дело жизнь кто первый очень два день её
новый рука даже во со раз где там под можно ну какой после их работа без самый потом надо хотеть ли
слово идти большой должен место иметь ничего видеть сейчас тоже стоять друг дом здесь теперь тут
думать сделать хорошо хороший спасибо пожалуйста привет пока здравствуй нормально отлично плохо
почему зачем сколько будет будешь будем буду есть нет да конечно может можешь могу хочу хочешь
люблю любишь знаешь знаю понимаю понимаешь помоги помочь скажи расскажи покажи напиши сегодня завтра
вчера утро вечер ночь неделя месяц деньги работаю учусь школа город страна мир россия москва столица
вопрос ответ правильно неправильно давай поговорим интересно скучно весело грустно игра музыка фильм
книга еда вода погода дождь снег солнце кот собака машина телефон компьютер интернет программа бот
нейросеть умный глупый тупой красивый быстро медленно много мало больше меньше лучше хуже всегда
никогда иногда часто тебя тебе меня мне нас вас вам нам ему ей них сделай делаешь делать зовут имя
""".split()
COMMON_WORDS = " ".join(COMMON_WORDS)

DONT_KNOW = [
    "Я пока не знаю, что ответить 🤔 Научи меня — нажми «✏️ Научить».",
    "Хм, этого я ещё не знаю. Покажешь правильный ответ? Нажми «✏️ Научить».",
    "Я ещё учусь и не понял 🙈 Можешь сказать иначе или научить меня?",
]


# ═══════════════════════════ ТЕКСТ И ОПЕЧАТКИ ═══════════════════════════

RU_LETTERS = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
LAYOUT_EN2RU = str.maketrans("qwertyuiop[]asdfghjkl;'zxcvbnm,.`", "йцукенгшщзхъфывапролджэячсмитьбюе")
WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.I)


def basic_norm(text: str) -> str:
    """Нижний регистр, ё->е, «привеееет» -> «привет», лишние пробелы."""
    t = text.lower().replace("ё", "е")
    t = re.sub(r"([^\d])\1{2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def contains_bad(text: str) -> bool:
    t = basic_norm(text)
    return any(b in t for b in BAD_WORDS)


def osa_distance(a: str, b: str) -> int:
    """Расстояние Дамерау-Левенштейна (с перестановками соседних букв)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


class SpellChecker:
    """Исправление опечаток по словарю (алгоритм SymSpell: индекс удалений букв)."""

    def __init__(self, max_words=30000):
        self.max_words = max_words
        self.freq = Counter()
        self.index = {}

    @staticmethod
    def _max_dist(word):
        return 1 if len(word) <= 4 else 2

    @staticmethod
    def _deletes(word, depth):
        out, frontier = {word}, {word}
        for _ in range(depth):
            nxt = set()
            for w in frontier:
                if len(w) > 1:
                    for i in range(len(w)):
                        nxt.add(w[:i] + w[i + 1:])
            out |= nxt
            frontier = nxt
        return out

    def build(self, word_counts: Counter):
        freq = Counter({w: c for w, c in word_counts.most_common(self.max_words)
                        if len(w) > 1 and not w.isdigit()})
        index = defaultdict(list)
        for w in freq:
            for d in self._deletes(w, self._max_dist(w)):
                index[d].append(w)
        self.freq, self.index = freq, dict(index)

    def add_words(self, words):
        for w in words:
            if len(w) > 1 and not w.isdigit() and w not in self.freq:
                for d in self._deletes(w, self._max_dist(w)):
                    self.index.setdefault(d, []).append(w)
            self.freq[w] += 1

    def correct_word(self, w: str) -> str:
        if len(w) <= 2 or w.isdigit() or w in self.freq:
            return w
        if re.fullmatch(r"[a-z]+", w):  # набрано в английской раскладке?
            ru = w.translate(LAYOUT_EN2RU)
            if ru in self.freq:
                return ru
            fixed = self.correct_word(ru)
            if fixed != ru and osa_distance(ru, fixed) <= 1:
                return fixed
            return w
        maxd = self._max_dist(w)
        best, best_key = w, None
        for d in self._deletes(w, maxd):
            for cand in self.index.get(d, ()):
                dist = osa_distance(w, cand)
                if dist <= maxd:
                    key = (dist, len(cand) < len(w), -self.freq[cand])  # чаще буквы пропускают
                    if best_key is None or key < best_key:
                        best, best_key = cand, key
        return best

    def correct(self, text: str) -> str:
        t = basic_norm(text)
        return WORD_RE.sub(lambda m: self.correct_word(m.group(0)), t)


# ═══════════════════════════ ТОКЕНИЗАТОР BPE ═══════════════════════════

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<usr>", "<bot>"]
PAD, UNK, BOS, EOS, USR, BOT = range(6)
BASE_ALPHABET = (RU_LETTERS + RU_LETTERS.upper() + "ёЁ" + "abcdefghijklmnopqrstuvwxyz"
                 + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789" + ".,!?-—:;()\"'«»…%+*/=@#№$&_<>[]{}~^|\\")
PRETOK_RE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\w\s]+| ?_+|\s+")


class BPETokenizer:
    """Byte-Pair Encoding: склеивает частые пары символов в токены (как в GPT)."""

    def __init__(self):
        self.vocab, self.tok2id, self.ranks = [], {}, {}
        self.cache = {}

    @property
    def size(self):
        return len(self.vocab)

    @staticmethod
    def _pretok(text):
        return [c.replace(" ", "▁") for c in PRETOK_RE.findall(re.sub(r"\s+", " ", text))]

    def train(self, texts, vocab_size):
        chunk_counts = Counter()
        for t in texts:
            chunk_counts.update(self._pretok(t))
        char_counts = Counter()
        for ch, c in chunk_counts.items():
            for x in ch:
                char_counts[x] += c
        alphabet = set(BASE_ALPHABET) | {"▁"} | {x for x, c in char_counts.items() if c >= 2}
        vocab = SPECIALS + sorted(alphabet)
        words = [list(w) for w in chunk_counts]
        freqs = list(chunk_counts.values())
        pair_counts, where = Counter(), defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[i]
                where[p].add(i)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        merges = []
        while len(vocab) < vocab_size and heap:
            negc, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair, 0)
            if cur <= 0:
                continue
            if cur != -negc:
                heapq.heappush(heap, (-cur, pair))
                continue
            if cur < 2:
                break
            new = pair[0] + pair[1]
            merges.append(pair)
            vocab.append(new)
            for i in list(where[pair]):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                nw, j = [], 0
                while j < len(w):
                    if j < len(w) - 1 and w[j] == pair[0] and w[j + 1] == pair[1]:
                        nw.append(new)
                        j += 2
                    else:
                        nw.append(w[j])
                        j += 1
                words[i] = nw
                for p in zip(nw, nw[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                    heapq.heappush(heap, (-pair_counts[p], p))
            del where[pair]
            pair_counts[pair] = 0
        self._set(vocab, merges)

    def _set(self, vocab, merges):
        self.vocab = list(dict.fromkeys(vocab))
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.ranks = {tuple(p): r for r, p in enumerate(merges)}
        self.cache = {}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": [list(p) for p in self.ranks]}, f, ensure_ascii=False)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self._set(d["vocab"], d["merges"])

    def _bpe(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        sym = list(chunk)
        while len(sym) > 1:
            best_i, best_r = None, None
            for i in range(len(sym) - 1):
                r = self.ranks.get((sym[i], sym[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_i, best_r = i, r
            if best_i is None:
                break
            a, b = sym[best_i], sym[best_i + 1]
            out, j = [], 0
            while j < len(sym):
                if j < len(sym) - 1 and sym[j] == a and sym[j + 1] == b:
                    out.append(a + b)
                    j += 2
                else:
                    out.append(sym[j])
                    j += 1
            sym = out
        ids = [self.tok2id.get(s, UNK) for s in sym]
        if len(self.cache) < 200_000:
            self.cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for chunk in self._pretok(text):
            ids.extend(self._bpe(chunk))
        return ids

    def decode(self, ids):
        toks = [self.vocab[i] for i in ids if i >= len(SPECIALS) and i < len(self.vocab)]
        return "".join(toks).replace("▁", " ").strip()


# ═══════════════════════════ НЕЙРОСЕТЬ GPT ═══════════════════════════

class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.resid_drop(self.proj(y))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Decoder-only трансформер: по предыдущим токенам предсказывает следующий."""

    def __init__(self, vocab, ctx, dim, layers, heads, dropout):
        super().__init__()
        self.cfg = dict(vocab=vocab, ctx=ctx, dim=dim, layers=layers, heads=heads, dropout=dropout)
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(ctx, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # общие веса входа и выхода
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=PAD)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new, temperature, top_k, top_p, rep_penalty, stop_ids):
        idx = torch.tensor([ids], dtype=torch.long)
        out = []
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.ctx:])
            logits = logits[0, -1].float()
            logits[PAD] = logits[UNK] = -float("inf")
            for t in set(out):
                logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < kth] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, 0)
                sp[cum - sp > top_p] = 0
                probs = torch.zeros_like(probs).scatter_(0, si, sp)
                probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1))
            if nxt in stop_ids:
                break
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        return out


# ═══════════════════════════ ПАМЯТЬ ОТВЕТОВ ═══════════════════════════

class Memory:
    """Поиск похожего вопроса: TF-IDF по буквенным триграммам + словам. Устойчив к опечаткам."""

    def __init__(self):
        self.items, self.inv, self.idf = [], {}, {}

    @staticmethod
    def _features(text):
        t = f" {text} "
        feats = Counter(t[i:i + 3] for i in range(len(t) - 2))
        for w in WORD_RE.findall(text):
            feats["w:" + w] += 2
        return feats

    def build(self, rows):
        """rows: (id, question, answer, weight)"""
        items, feats_all, df = [], [], Counter()
        for qa_id, q, a, w in rows:
            f = self._features(q)
            items.append({"id": qa_id, "q": q, "a": a, "w": max(w, 0.1)})
            feats_all.append(f)
            df.update(f.keys())
        n = len(items)
        idf = {k: math.log((n + 1) / (c + 1)) + 1 for k, c in df.items()}
        inv = defaultdict(list)
        for i, f in enumerate(feats_all):
            vec = {k: v * idf[k] for k, v in f.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1
            for k, v in vec.items():
                inv[k].append((i, v / norm))
        self.items, self.inv, self.idf = items, dict(inv), idf

    def search(self, text, top=10):
        top_idf = max(self.idf.values(), default=1.0)  # незнакомые признаки — как самые редкие
        vec = {k: v * self.idf.get(k, top_idf) for k, v in self._features(text).items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1
        scores = defaultdict(float)
        for k, v in vec.items():
            for i, w in self.inv.get(k, ()):
                scores[i] += v / norm * w
        best = sorted(scores.items(), key=lambda x: -x[1])[:top]
        return [(s, self.items[i]) for i, s in best]

    @staticmethod
    def pick(hits):
        best = hits[0][0]
        pool = [it for s, it in hits if s >= best - 0.05]
        return random.choices(pool, weights=[it["w"] for it in pool])[0]


# ═══════════════════════════ БАЗА ДАННЫХ ═══════════════════════════

class DB:
    def __init__(self, path):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.x("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT, name TEXT,
                  joined REAL, sub_until REAL DEFAULT 0, day TEXT, today INTEGER DEFAULT 0,
                  total INTEGER DEFAULT 0, banned INTEGER DEFAULT 0)""")
        self.x("""CREATE TABLE IF NOT EXISTS qa(id INTEGER PRIMARY KEY AUTOINCREMENT, q TEXT, a TEXT,
                  weight REAL DEFAULT 1, approved INTEGER DEFAULT 1, author INTEGER DEFAULT 0, created REAL)""")
        self.x("CREATE TABLE IF NOT EXISTS texts(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, created REAL)")
        self.x("""CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
                  q TEXT, a TEXT, source TEXT, qa_id INTEGER, voted INTEGER DEFAULT 0, created REAL)""")
        self.x("""CREATE TABLE IF NOT EXISTS payments(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
                  plan TEXT, stars INTEGER, charge_id TEXT, created REAL)""")
        self.x("CREATE TABLE IF NOT EXISTS support(admin_chat INTEGER, admin_msg INTEGER, user_id INTEGER)")
        self.x("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")

    def x(self, sql, args=()):
        with self.lock:
            cur = self.con.execute(sql, args)
            self.con.commit()
            return cur

    def one(self, sql, args=()):
        with self.lock:
            return self.con.execute(sql, args).fetchone()

    def all(self, sql, args=()):
        with self.lock:
            return self.con.execute(sql, args).fetchall()

    def meta(self, k, default=None):
        r = self.one("SELECT v FROM meta WHERE k=?", (k,))
        return r["v"] if r else default

    def set_meta(self, k, v):
        self.x("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, str(v)))

    # пользователи
    def user(self, u):
        self.x("INSERT OR IGNORE INTO users(id, username, name, joined) VALUES(?,?,?,?)",
               (u.id, u.username, u.full_name, time.time()))
        self.x("UPDATE users SET username=?, name=? WHERE id=?", (u.username, u.full_name, u.id))
        return self.one("SELECT * FROM users WHERE id=?", (u.id,))

    def has_sub(self, uid):
        r = self.one("SELECT sub_until FROM users WHERE id=?", (uid,))
        return bool(r and r["sub_until"] > time.time())

    def extend_sub(self, uid, days):
        r = self.one("SELECT sub_until FROM users WHERE id=?", (uid,))
        start = max(time.time(), r["sub_until"] if r else 0)
        until = start + days * 86400
        self.x("UPDATE users SET sub_until=? WHERE id=?", (until, uid))
        return until

    def count_message(self, uid):
        """Учитывает сообщение. Возвращает сколько сообщений уже сегодня."""
        today = datetime.now().strftime("%Y-%m-%d")
        r = self.one("SELECT day, today FROM users WHERE id=?", (uid,))
        n = (r["today"] if r and r["day"] == today else 0) + 1
        self.x("UPDATE users SET day=?, today=?, total=total+1 WHERE id=?", (today, n, uid))
        return n

    def used_today(self, uid):
        r = self.one("SELECT day, today FROM users WHERE id=?", (uid,))
        return r["today"] if r and r["day"] == datetime.now().strftime("%Y-%m-%d") else 0

    # знания
    def add_qa(self, q, a, author=0, weight=1.0, approved=1):
        return self.x("INSERT INTO qa(q,a,weight,approved,author,created) VALUES(?,?,?,?,?,?)",
                      (q, a, weight, approved, author, time.time())).lastrowid

    def find_qa(self, q, a):
        return self.one("SELECT * FROM qa WHERE q=? AND a=?", (q, a))

    def active_qa(self):
        return self.all("SELECT id,q,a,weight FROM qa WHERE approved=1 AND weight>0")

    def add_text(self, text):
        self.x("INSERT INTO texts(text, created) VALUES(?,?)", (text, time.time()))

    def new_examples_since(self, ts):
        a = self.one("SELECT COUNT(*) c FROM qa WHERE created>? AND approved=1", (ts,))["c"]
        b = self.one("SELECT COUNT(*) c FROM texts WHERE created>?", (ts,))["c"]
        return a + b


# ═══════════════════════════ МОЗГ (всё вместе) ═══════════════════════════

class Brain:
    def __init__(self, db: DB):
        self.db = db
        self.tok = BPETokenizer()
        self.spell = SpellChecker()
        self.memory = Memory()
        self.model = None                  # модель для ответов (CPU)
        self.train_model = None            # копия для обучения (GPU/MPS/CPU)
        self.optimizer = None
        self.lock = threading.Lock()
        self.train_now = threading.Event()
        self.steps = 0
        self.loss = None
        self.status = "запуск"
        self.corpus_cache = {}
        if DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = DEVICE

    # ---------- данные ----------
    def seed(self):
        if self.db.one("SELECT COUNT(*) c FROM qa")["c"]:
            return
        for qs, ans in SEED_DIALOGS:
            for q in qs:
                for a in ans:
                    self.db.add_qa(basic_norm(q), a)
        log("Загружены стартовые диалоги")

    def corpus_texts(self):
        """Тексты из data/corpus/*.txt (кешируются)."""
        out, total = [], 0
        for name in sorted(os.listdir(CORPUS_DIR)):
            if not name.lower().endswith(".txt"):
                continue
            path = os.path.join(CORPUS_DIR, name)
            key = (path, os.path.getmtime(path))
            if key not in self.corpus_cache:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    raw = f.read(CORPUS_MAX_CHARS)
                paras = [p.strip() for p in re.split(r"\n\s*\n|\n", raw) if len(p.strip()) > 2]
                self.corpus_cache[key] = paras
            for p in self.corpus_cache[key]:
                if total > CORPUS_MAX_CHARS:
                    return out
                out.append(p)
                total += len(p)
        return out

    def rebuild_knowledge(self):
        rows = self.db.active_qa()
        self.memory.build([(r["id"], r["q"], r["a"], r["weight"]) for r in rows])
        words = Counter({w: 1 for w in COMMON_WORDS.split()})
        for r in rows:
            words.update(WORD_RE.findall(basic_norm(r["q"])))
            words.update(w for w in WORD_RE.findall(basic_norm(r["a"])))
        for p in self.corpus_texts()[:200_000]:
            words.update(WORD_RE.findall(basic_norm(p)))
        user_words = Counter()
        for r in self.db.all("SELECT text FROM texts ORDER BY id DESC LIMIT 50000"):
            user_words.update(WORD_RE.findall(r["text"]))
        words.update({w: c for w, c in user_words.items() if c >= 3})  # опечатки пользователей не учим
        self.spell.build(words)

    def build_stream(self):
        """Собирает все данные в одну последовательность токенов для обучения."""
        ids = []
        for r in self.db.active_qa():
            ex = [BOS, USR] + self.tok.encode(" " + r["q"]) + [BOT] + self.tok.encode(" " + r["a"]) + [EOS]
            ids.extend(ex * max(1, min(int(round(r["weight"])), 5)))
        for r in self.db.all("SELECT text FROM texts ORDER BY id DESC LIMIT 100000"):
            ids.extend([BOS] + self.tok.encode(" " + r["text"]) + [EOS])
        for p in self.corpus_texts():
            ids.extend([BOS] + self.tok.encode(" " + p) + [EOS])
        return torch.tensor(ids, dtype=torch.long)

    # ---------- инициализация ----------
    def init(self):
        self.seed()
        if os.path.exists(TOKENIZER_PATH):
            self.tok.load(TOKENIZER_PATH)
        else:
            log("Обучаю токенизатор…")
            texts = [r["q"] + " " + r["a"] for r in self.db.active_qa()]
            texts += [r["text"] for r in self.db.all("SELECT text FROM texts")]
            texts += self.corpus_texts()[:300_000]
            self.tok.train(texts, VOCAB_SIZE)
            self.tok.save(TOKENIZER_PATH)
            log(f"Токенизатор готов: {self.tok.size} токенов")
        if os.path.exists(MODEL_PATH):
            ck = torch.load(MODEL_PATH, map_location="cpu")
            self.model = GPT(**ck["cfg"])
            self.model.load_state_dict(ck["model"])
            self.steps, self.loss = ck.get("steps", 0), ck.get("loss")
            log(f"Нейросеть загружена: {self.steps} шагов обучения, loss={self.loss}")
        else:
            self.model = GPT(self.tok.size, CONTEXT_LEN, MODEL_DIM, MODEL_LAYERS, MODEL_HEADS, DROPOUT)
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        log(f"Параметров в нейросети: {n / 1e6:.2f} млн, обучение на: {self.device}")
        self.rebuild_knowledge()
        self.status = "готов"
        threading.Thread(target=self._trainer_loop, daemon=True).start()

    # ---------- обучение ----------
    def _trainer_loop(self):
        if self.steps == 0:
            self.train(FIRST_TRAIN_STEPS)
        while True:
            forced = self.train_now.wait(RETRAIN_EVERY_MIN * 60)
            self.train_now.clear()
            try:
                last = float(self.db.meta("last_train", 0))
                if forced or self.db.new_examples_since(last) >= RETRAIN_MIN_NEW:
                    self.rebuild_knowledge()
                    self.train(RETRAIN_STEPS)
            except Exception as e:
                log("Ошибка обучения:", e)
                self.status = f"ошибка: {e}"

    def train(self, steps):
        started = time.time()
        data = self.build_stream()
        ctx = self.model.ctx
        if len(data) < ctx + 2:
            log("Мало данных для обучения")
            return
        if self.train_model is None:
            self.train_model = copy.deepcopy(self.model).to(self.device)
            self.optimizer = torch.optim.AdamW(self.train_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        m = self.train_model
        m.train()
        log(f"Обучение: {steps} шагов на {len(data)} токенах")
        losses, published = [], 0
        for step in range(1, steps + 1):
            lr = LEARNING_RATE * min(1.0, step / 100)  # плавный разгон
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            ix = torch.randint(0, len(data) - ctx - 1, (BATCH_SIZE,))
            x = torch.stack([data[i:i + ctx] for i in ix]).to(self.device)
            y = torch.stack([data[i + 1:i + ctx + 1] for i in ix]).to(self.device)
            _, loss = m(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())
            if step % 50 == 0 or step == steps:
                self.loss = round(sum(losses[-50:]) / len(losses[-50:]), 3)
                self.status = f"обучение {step}/{steps}, loss {self.loss}"
            if step % 250 == 0 or step == steps:
                self._publish(step - published)
                published = step
        m.eval()
        self.status = "готов"
        self.db.set_meta("last_train", started)
        log(f"Обучение завершено за {time.time() - started:.0f} сек, loss={self.loss}, всего шагов {self.steps}")

    def _publish(self, new_steps):
        """Переносит веса из обучаемой копии в рабочую модель и сохраняет на диск."""
        with self.lock:
            self.model.load_state_dict({k: v.detach().cpu() for k, v in self.train_model.state_dict().items()})
            self.steps += new_steps
            state = {k: v.clone() for k, v in self.model.state_dict().items()}
        tmp = MODEL_PATH + ".tmp"
        torch.save({"model": state, "cfg": self.model.cfg, "steps": self.steps, "loss": self.loss}, tmp)
        os.replace(tmp, MODEL_PATH)

    # ---------- ответы ----------
    @staticmethod
    def skills(text):
        if re.search(r"\b(сколько времени|который час|время сейчас|текущее время)\b", text):
            return f"Сейчас {datetime.now().strftime('%H:%M')} ⏰"
        if re.search(r"\b(какое сегодня число|какая сегодня дата|какое число|сегодняшняя дата)\b", text):
            return f"Сегодня {datetime.now().strftime('%d.%m.%Y')} 📅"
        expr = re.sub(r"[^\d+\-*/().,^ ]", "", text.replace("х", "*").replace("x", "*")).replace(",", ".").strip()
        if re.search(r"\d\s*[+\-*/^]\s*[\d(]", expr):
            val = safe_eval(expr.replace("^", "**"))
            if val is not None:
                return f"{expr.replace('**', '^')} = {val}"
        return None

    def model_ready(self):
        return self.steps >= MIN_STEPS_TO_SPEAK

    def generate(self, text, history):
        ids = []
        for q, a in list(history)[-HISTORY_TURNS:]:
            ids += [BOS, USR] + self.tok.encode(" " + q) + [BOT] + self.tok.encode(" " + a) + [EOS]
        ids += [BOS, USR] + self.tok.encode(" " + text) + [BOT]
        ids = ids[-(self.model.ctx - MAX_ANSWER_TOKENS):]
        with self.lock:
            out = self.model.generate(ids, MAX_ANSWER_TOKENS, TEMPERATURE, TOP_K, TOP_P,
                                      REPETITION_PENALTY, {EOS, USR, BOS, PAD})
        ans = self.tok.decode(out).strip()
        return ans[:1].upper() + ans[1:]

    @staticmethod
    def looks_ok(ans):
        letters = sum(ch.isalpha() for ch in ans)
        return len(ans) >= 2 and (letters >= 2 or any(ord(c) > 0x2000 for c in ans))

    def reply(self, text, history):
        """Возвращает (ответ, источник, исправленный_текст, qa_id)."""
        corrected = self.spell.correct(text)
        sk = self.skills(corrected)
        if sk:
            return sk, "skill", corrected, None
        hits = self.memory.search(corrected) if self.memory.items else []
        strong = hits and hits[0][0] >= MEMORY_THRESHOLD
        if strong and ANSWER_MODE == "hybrid":
            it = Memory.pick(hits)
            return it["a"], "memory", corrected, it["id"]
        if self.model_ready():
            for _ in range(2):
                ans = self.generate(corrected, history)
                if self.looks_ok(ans):
                    return ans, "neural", corrected, None
        if hits and hits[0][0] >= MEMORY_THRESHOLD * 0.6:
            it = Memory.pick(hits)
            return it["a"], "memory", corrected, it["id"]
        return random.choice(DONT_KNOW), "unknown", corrected, None

    def learn_pair(self, q, a, author, weight=1.0):
        """Учит пару вопрос->ответ. Возвращает (qa_id, нужна_модерация)."""
        q = self.spell.correct(q)
        if not (MODERATION and author not in ADMIN_IDS):
            self.spell.add_words(WORD_RE.findall(q) + WORD_RE.findall(basic_norm(a)))
        a = a.strip()
        need_mod = MODERATION and author not in ADMIN_IDS
        old = self.db.find_qa(q, a)
        if old:
            self.db.x("UPDATE qa SET weight=weight+? WHERE id=?", (weight, old["id"]))
            qa_id = old["id"]
        else:
            qa_id = self.db.add_qa(q, a, author=author, weight=weight, approved=0 if need_mod else 1)
        if not need_mod:
            self.rebuild_memory_only()
        return qa_id, need_mod

    def rebuild_memory_only(self):
        rows = self.db.active_qa()
        self.memory.build([(r["id"], r["q"], r["a"], r["weight"]) for r in rows])


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr):
    """Безопасный калькулятор (без eval)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 100:
                raise ValueError
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        v = ev(ast.parse(expr, mode="eval").body)
        if isinstance(v, float):
            v = round(v, 6)
            if v.is_integer():
                v = int(v)
        return v
    except Exception:
        return None


# ═══════════════════════════ TELEGRAM БОТ ═══════════════════════════

db = DB(DB_PATH)
brain = Brain(db)
router = Router()

BTN_CHAT, BTN_SUB, BTN_SUPPORT, BTN_PROFILE, BTN_TEACH = (
    "💬 Чат", "⭐ Подписка", "🆘 Поддержка", "👤 Профиль", "🧠 Научить")
MENU = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
    [KeyboardButton(text=BTN_CHAT), KeyboardButton(text=BTN_SUB)],
    [KeyboardButton(text=BTN_SUPPORT), KeyboardButton(text=BTN_PROFILE)],
    [KeyboardButton(text=BTN_TEACH)],
])

states = {}                                              # user_id -> (состояние, данные)
histories = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))  # контекст диалога
last_msg_time = {}


def is_admin(uid):
    return uid in ADMIN_IDS


def fmt_date(ts):
    return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")


def feedback_kb(log_id):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="👍", callback_data=f"fb:like:{log_id}"),
        InlineKeyboardButton(text="👎", callback_data=f"fb:dis:{log_id}"),
        InlineKeyboardButton(text="✏️ Научить", callback_data=f"fb:teach:{log_id}"),
    ]])


def plans_kb():
    rows = [[InlineKeyboardButton(text=f"{p['title']} — {p['stars']} ⭐", callback_data=f"buy:{k}")]
            for k, p in SUBSCRIPTION_PLANS.items()]
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def notify_admins(bot: Bot, text, **kw):
    for a in ADMIN_IDS:
        try:
            await bot.send_message(a, text, **kw)
        except Exception:
            pass


# ---------- команды ----------

@router.message(CommandStart())
async def cmd_start(m: Message):
    db.user(m.from_user)
    states.pop(m.from_user.id, None)
    await m.answer(
        f"Привет, {html.escape(m.from_user.first_name or 'друг')}! 👋\n\n"
        f"Я <b>{BOT_NAME}</b> — собственная нейросеть. Я понимаю слова даже с опечатками "
        f"и учусь у каждого собеседника 🧠\n\n"
        f"Просто напиши мне что-нибудь. Если я ответил плохо — нажми «✏️ Научить» под ответом.\n\n"
        f"Бесплатно: {FREE_MESSAGES_PER_DAY} сообщений в день, с подпиской ⭐ — без ограничений.",
        reply_markup=MENU)


@router.message(Command("help"))
async def cmd_help(m: Message):
    text = ("<b>Команды</b>\n/new — начать диалог заново\n/teach вопрос = ответ — научить меня\n"
            "/profile — профиль\n/paysupport — помощь с оплатой\n/terms — условия")
    if is_admin(m.from_user.id):
        text += ("\n\n<b>Админ</b>\n/admin — панель\n/train — обучить сейчас\n/give ID дни — выдать подписку\n"
                 "/ban ID, /unban ID\n/broadcast текст — рассылка\n/refund ID charge_id — вернуть звёзды\n"
                 "Отправь .txt файл — добавлю в корпус для обучения")
    await m.answer(text, reply_markup=MENU)


@router.message(Command("new"))
async def cmd_new(m: Message):
    histories.pop(m.from_user.id, None)
    states.pop(m.from_user.id, None)
    await m.answer("🔄 Начинаем новый диалог!", reply_markup=MENU)


@router.message(Command("terms"))
async def cmd_terms(m: Message):
    await m.answer("📄 <b>Условия</b>\nПодписка даёт безлимитное общение с ботом на выбранный срок. "
                   "Оплата — Telegram Stars. Бот — обучающаяся нейросеть и может ошибаться. "
                   "Вопросы по оплате: /paysupport")


@router.message(Command("paysupport"))
async def cmd_paysupport(m: Message):
    states[m.from_user.id] = ("support", None)
    await m.answer("💳 Опиши проблему с оплатой одним сообщением — передам админу.")


@router.message(Command("profile"))
@router.message(Filt.text == BTN_PROFILE)
async def cmd_profile(m: Message):
    u = db.user(m.from_user)
    uid = m.from_user.id
    if is_admin(uid):
        sub = "👑 админ (безлимит)"
    elif u["sub_until"] > time.time():
        sub = f"⭐ активна до {fmt_date(u['sub_until'])}"
    else:
        sub = "нет"
    taught = db.one("SELECT COUNT(*) c FROM qa WHERE author=?", (uid,))["c"]
    await m.answer(f"👤 <b>Профиль</b>\nID: <code>{uid}</code>\nПодписка: {sub}\n"
                   f"Сегодня сообщений: {db.used_today(uid)}"
                   f"{'' if db.has_sub(uid) or is_admin(uid) else f' из {FREE_MESSAGES_PER_DAY}'}\n"
                   f"Всего сообщений: {u['total']}\nНаучил меня ответам: {taught} 🧠", reply_markup=MENU)


@router.message(Command("teach"))
async def cmd_teach(m: Message):
    arg = m.text.partition(" ")[2]
    if "=" not in arg:
        states[m.from_user.id] = ("teach_q", None)
        await m.answer("🧠 Напиши <b>вопрос</b> (что мне могут написать):")
        return
    q, _, a = arg.partition("=")
    await do_teach(m, q.strip(), a.strip())


async def do_teach(m: Message, q, a):
    uid = m.from_user.id
    if len(q) < 1 or len(a) < 1 or len(a) > 1000:
        await m.answer("Вопрос и ответ не должны быть пустыми (ответ до 1000 символов).")
        return
    if contains_bad(q) or contains_bad(a):
        await m.answer("🚫 Этому я учиться не буду.")
        return
    qa_id, need_mod = await asyncio.to_thread(brain.learn_pair, q, a, uid, 2.0)
    if need_mod:
        await m.answer("✅ Спасибо! Ответ отправлен на проверку админу.", reply_markup=MENU)
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✅ Принять", callback_data=f"mod:ok:{qa_id}"),
            InlineKeyboardButton(text="❌ Отклонить", callback_data=f"mod:no:{qa_id}")]])
        await notify_admins(m.bot, f"📝 Новое обучение от <code>{uid}</code>\n<b>Вопрос:</b> {html.escape(q)}\n"
                                   f"<b>Ответ:</b> {html.escape(a)}", reply_markup=kb)
    else:
        await m.answer("✅ Запомнил! Теперь буду так отвечать. Нейросеть дообучится в фоне 🧠", reply_markup=MENU)


# ---------- админ ----------

def admin_stats():
    users = db.one("SELECT COUNT(*) c FROM users")["c"]
    subs = db.one("SELECT COUNT(*) c FROM users WHERE sub_until>?", (time.time(),))["c"]
    stars = db.one("SELECT COALESCE(SUM(stars),0) s FROM payments")["s"]
    qa = db.one("SELECT COUNT(*) c FROM qa WHERE approved=1")["c"]
    pend = db.one("SELECT COUNT(*) c FROM qa WHERE approved=0")["c"]
    texts = db.one("SELECT COUNT(*) c FROM texts")["c"]
    msgs = db.one("SELECT COUNT(*) c FROM logs")["c"]
    likes = db.one("SELECT COUNT(*) c FROM logs WHERE voted=1")["c"]
    dis = db.one("SELECT COUNT(*) c FROM logs WHERE voted=-1")["c"]
    src = {r["source"]: r["c"] for r in db.all("SELECT source, COUNT(*) c FROM logs GROUP BY source")}
    n_par = sum(p.numel() for p in brain.model.parameters()) / 1e6
    return (f"🛠 <b>Админ-панель</b>\n\n👥 Пользователей: {users}\n⭐ Подписчиков: {subs}\n"
            f"💰 Получено звёзд: {stars}\n\n💬 Сообщений: {msgs} (👍 {likes} / 👎 {dis})\n"
            f"Источники ответов: нейросеть {src.get('neural', 0)}, память {src.get('memory', 0)}, "
            f"навыки {src.get('skill', 0)}, не знал {src.get('unknown', 0)}\n\n"
            f"🧠 <b>Нейросеть</b>: {n_par:.2f} млн параметров, {brain.tok.size} токенов в словаре\n"
            f"Шагов обучения: {brain.steps}, loss: {brain.loss}\nСтатус: {brain.status}\n"
            f"Устройство: {brain.device}\n\n📚 Пар вопрос-ответ: {qa} (на модерации: {pend})\n"
            f"Текстов от пользователей: {texts}\nФайлов в корпусе: "
            f"{len([f for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')])}")


def admin_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🧠 Обучить сейчас", callback_data="adm:train"),
         InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:stats")],
        [InlineKeyboardButton(text="📝 Модерация", callback_data="adm:mod")],
    ])


@router.message(Command("admin"))
async def cmd_admin(m: Message):
    if is_admin(m.from_user.id):
        await m.answer(admin_stats(), reply_markup=admin_kb())


@router.message(Command("train"))
async def cmd_train(m: Message):
    if is_admin(m.from_user.id):
        brain.train_now.set()
        await m.answer("🧠 Запускаю дообучение в фоне. Статус — в /admin")


@router.message(Command("give"))
async def cmd_give(m: Message):
    if not is_admin(m.from_user.id):
        return
    try:
        _, uid, days = m.text.split()
        until = db.extend_sub(int(uid), int(days))
        await m.answer(f"✅ Подписка пользователю {uid} до {fmt_date(until)}")
        await m.bot.send_message(int(uid), f"🎁 Тебе выдана подписка до {fmt_date(until)}!")
    except Exception as e:
        await m.answer(f"Формат: /give ID дни\n{e}")


@router.message(Command("ban", "unban"))
async def cmd_ban(m: Message):
    if not is_admin(m.from_user.id):
        return
    try:
        cmd, uid = m.text.split()
        db.x("UPDATE users SET banned=? WHERE id=?", (1 if cmd == "/ban" else 0, int(uid)))
        await m.answer("✅ Готово")
    except Exception:
        await m.answer("Формат: /ban ID или /unban ID")


@router.message(Command("broadcast"))
async def cmd_broadcast(m: Message):
    if not is_admin(m.from_user.id):
        return
    text = m.text.partition(" ")[2].strip()
    if not text:
        await m.answer("Формат: /broadcast текст")
        return
    ok = fail = 0
    for r in db.all("SELECT id FROM users WHERE banned=0"):
        try:
            await m.bot.send_message(r["id"], text)
            ok += 1
        except Exception:
            fail += 1
        await asyncio.sleep(0.05)
    await m.answer(f"📢 Рассылка: доставлено {ok}, ошибок {fail}")


@router.message(Command("refund"))
async def cmd_refund(m: Message):
    if not is_admin(m.from_user.id):
        return
    try:
        _, uid, charge = m.text.split()
        await m.bot.refund_star_payment(user_id=int(uid), telegram_payment_charge_id=charge)
        db.x("UPDATE users SET sub_until=0 WHERE id=?", (int(uid),))
        await m.answer("✅ Звёзды возвращены, подписка отключена")
    except Exception as e:
        await m.answer(f"Формат: /refund ID charge_id\n{e}")


@router.callback_query(Filt.data.startswith("adm:"))
async def cb_admin(c: CallbackQuery):
    if not is_admin(c.from_user.id):
        return await c.answer()
    act = c.data.split(":")[1]
    if act == "train":
        brain.train_now.set()
        await c.answer("Обучение запущено 🧠")
    elif act == "stats":
        try:
            await c.message.edit_text(admin_stats(), reply_markup=admin_kb())
        except Exception:
            pass
        await c.answer()
    elif act == "mod":
        await c.answer()
        await send_next_moderation(c.message)


async def send_next_moderation(msg: Message):
    r = db.one("SELECT * FROM qa WHERE approved=0 ORDER BY id LIMIT 1")
    if not r:
        await msg.answer("📝 На модерации ничего нет")
        return
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Принять", callback_data=f"mod:ok:{r['id']}"),
        InlineKeyboardButton(text="❌ Отклонить", callback_data=f"mod:no:{r['id']}")]])
    await msg.answer(f"📝 От <code>{r['author']}</code>\n<b>Вопрос:</b> {html.escape(r['q'])}\n"
                     f"<b>Ответ:</b> {html.escape(r['a'])}", reply_markup=kb)


@router.callback_query(Filt.data.startswith("mod:"))
async def cb_mod(c: CallbackQuery):
    if not is_admin(c.from_user.id):
        return await c.answer()
    _, act, qa_id = c.data.split(":")
    if act == "ok":
        db.x("UPDATE qa SET approved=1 WHERE id=?", (int(qa_id),))
        await asyncio.to_thread(brain.rebuild_memory_only)
    else:
        db.x("DELETE FROM qa WHERE id=? AND approved=0", (int(qa_id),))
    await c.answer("Принято ✅" if act == "ok" else "Отклонено ❌")
    try:
        await c.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await send_next_moderation(c.message)


@router.message(Filt.document, lambda m: is_admin(m.from_user.id))
async def admin_document(m: Message):
    name = m.document.file_name or "file.txt"
    if not name.lower().endswith(".txt"):
        await m.answer("Принимаю только .txt файлы для обучения")
        return
    path = os.path.join(CORPUS_DIR, f"{int(time.time())}_{re.sub(r'[^\w.-]', '_', name)}")
    await m.bot.download(m.document, destination=path)
    brain.train_now.set()
    await m.answer(f"📚 Файл добавлен в корпус. Запускаю дообучение в фоне.\n"
                   f"⚠️ Новые слова из файла попадут в словарь токенов только после удаления "
                   f"data/tokenizer.json и data/model.pt (переобучение с нуля).")


# ---------- поддержка: ответ админа ----------

def support_target(m: Message):
    if not (m.reply_to_message and is_admin(m.from_user.id)):
        return None
    r = db.one("SELECT user_id FROM support WHERE admin_chat=? AND admin_msg=?",
               (m.chat.id, m.reply_to_message.message_id))
    return r["user_id"] if r else None


@router.message(lambda m: support_target(m) is not None)
async def admin_support_reply(m: Message):
    uid = support_target(m)
    try:
        await m.bot.send_message(uid, "💬 <b>Ответ поддержки:</b>")
        await m.copy_to(uid)
        await m.answer("✅ Ответ отправлен")
    except Exception as e:
        await m.answer(f"Не удалось отправить: {e}")


# ---------- кнопки меню ----------

@router.message(Filt.text == BTN_CHAT)
async def btn_chat(m: Message):
    states.pop(m.from_user.id, None)
    await m.answer("💬 Пиши мне что угодно — я отвечу! Чтобы начать заново: /new", reply_markup=MENU)


@router.message(Filt.text == BTN_SUB)
async def btn_sub(m: Message):
    uid = m.from_user.id
    status = (f"✅ Твоя подписка активна до {fmt_date(db.one('SELECT sub_until FROM users WHERE id=?', (uid,))['sub_until'])}\n\n"
              if db.has_sub(uid) else "")
    await m.answer(f"⭐ <b>Подписка</b>\n\n{status}С подпиской — безлимитное общение с нейросетью и "
                   f"поддержка развития бота.\nБез подписки: {FREE_MESSAGES_PER_DAY} сообщений в день.\n\n"
                   f"Выбери тариф (оплата Telegram Stars):", reply_markup=plans_kb())


@router.message(Filt.text == BTN_SUPPORT)
async def btn_support(m: Message):
    states[m.from_user.id] = ("support", None)
    await m.answer("🆘 Напиши своё сообщение для поддержки (можно с фото). Админ ответит сюда же.\n"
                   "Передумал — нажми «💬 Чат».")


@router.message(Filt.text == BTN_TEACH)
async def btn_teach(m: Message):
    states[m.from_user.id] = ("teach_q", None)
    await m.answer("🧠 Научи меня! Напиши <b>вопрос</b> (что мне могут написать):")


# ---------- оплата Telegram Stars ----------

@router.callback_query(Filt.data.startswith("buy:"))
async def cb_buy(c: CallbackQuery):
    key = c.data.split(":")[1]
    p = SUBSCRIPTION_PLANS.get(key)
    if not p:
        return await c.answer("Тариф не найден")
    await c.bot.send_invoice(
        chat_id=c.from_user.id, title=f"Подписка «{p['title']}»",
        description=f"Безлимитное общение с {BOT_NAME} на {p['days']} дн.",
        payload=f"sub:{key}", provider_token="", currency="XTR",
        prices=[LabeledPrice(label=f"Подписка {p['title']}", amount=p["stars"])])
    await c.answer()


@router.pre_checkout_query()
async def pre_checkout(q: PreCheckoutQuery):
    key = q.invoice_payload.split(":")[-1]
    p = SUBSCRIPTION_PLANS.get(key)
    if p and q.currency == "XTR" and q.total_amount == p["stars"]:
        await q.answer(ok=True)
    else:
        await q.answer(ok=False, error_message="Тариф изменился, попробуй ещё раз.")


@router.message(Filt.successful_payment)
async def paid(m: Message):
    sp = m.successful_payment
    key = sp.invoice_payload.split(":")[-1]
    p = SUBSCRIPTION_PLANS.get(key, {"days": 30, "title": key})
    db.user(m.from_user)
    until = db.extend_sub(m.from_user.id, p["days"])
    db.x("INSERT INTO payments(user_id, plan, stars, charge_id, created) VALUES(?,?,?,?,?)",
         (m.from_user.id, key, sp.total_amount, sp.telegram_payment_charge_id, time.time()))
    await m.answer(f"🎉 Спасибо! Подписка активна до <b>{fmt_date(until)}</b>", reply_markup=MENU)
    await notify_admins(m.bot, f"💰 Оплата {sp.total_amount} ⭐ от <code>{m.from_user.id}</code> "
                               f"(@{m.from_user.username}), тариф {p['title']}\n"
                               f"charge_id: <code>{sp.telegram_payment_charge_id}</code>")


# ---------- обратная связь под ответами ----------

@router.callback_query(Filt.data.startswith("fb:"))
async def cb_feedback(c: CallbackQuery):
    _, act, log_id = c.data.split(":")
    r = db.one("SELECT * FROM logs WHERE id=? AND user_id=?", (int(log_id), c.from_user.id))
    if not r:
        return await c.answer()
    if act == "teach":
        states[c.from_user.id] = ("teach_a", r["q"])
        await c.answer()
        await c.message.answer(f"✏️ Как правильно ответить на «{html.escape(r['q'])}»? Напиши ответ:")
        return
    if r["voted"]:
        return await c.answer("Ты уже оценил этот ответ")
    if act == "like":
        db.x("UPDATE logs SET voted=1 WHERE id=?", (r["id"],))
        if r["source"] in ("neural", "memory") and LEARN_FROM_USERS and not contains_bad(r["a"]):
            await asyncio.to_thread(brain.learn_pair, r["q"], r["a"], c.from_user.id, 1.0)
        await c.answer("Спасибо! Запомню, что это хороший ответ 👍")
    else:
        db.x("UPDATE logs SET voted=-1 WHERE id=?", (r["id"],))
        if r["qa_id"]:
            db.x("UPDATE qa SET weight=weight-0.5 WHERE id=?", (r["qa_id"],))
            await asyncio.to_thread(brain.rebuild_memory_only)
        await c.answer("Понял, буду отвечать иначе. Нажми «✏️ Научить», чтобы показать правильный ответ")
    try:
        await c.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✏️ Научить", callback_data=f"fb:teach:{log_id}")]]))
    except Exception:
        pass


# ---------- основной обработчик сообщений ----------

async def forward_to_support(m: Message):
    for a in ADMIN_IDS:
        try:
            head = await m.bot.send_message(
                a, f"🆘 Обращение от <code>{m.from_user.id}</code> "
                   f"(@{m.from_user.username or '—'}, {html.escape(m.from_user.full_name)})\n"
                   f"Ответь <b>реплаем</b> на сообщение ниже — ответ уйдёт пользователю.")
            copied = await m.copy_to(a)
            for mid in (head.message_id, copied.message_id):
                db.x("INSERT INTO support(admin_chat, admin_msg, user_id) VALUES(?,?,?)", (a, mid, m.from_user.id))
        except Exception as e:
            log("Поддержка: не удалось отправить админу", a, e)
    await m.answer("✅ Сообщение отправлено в поддержку. Ответ придёт сюда.", reply_markup=MENU)


@router.message(Filt.text)
async def on_text(m: Message):
    uid = m.from_user.id
    u = db.user(m.from_user)
    if u["banned"]:
        return
    text = m.text.strip()

    st = states.pop(uid, None)
    if st:
        kind, data = st
        if kind == "support":
            return await forward_to_support(m)
        if kind == "teach_q":
            states[uid] = ("teach_a", text)
            return await m.answer(f"Отлично! Теперь напиши <b>ответ</b> на «{html.escape(text)}»:")
        if kind == "teach_a":
            return await do_teach(m, data, text)

    if text.startswith("/"):
        return await m.answer("Не знаю такой команды. Список: /help")
    if len(text) > 2000:
        return await m.answer("Слишком длинное сообщение 🙈 Напиши короче (до 2000 символов).")

    now = time.time()
    if now - last_msg_time.get(uid, 0) < FLOOD_DELAY_SEC:
        return
    last_msg_time[uid] = now

    unlimited = is_admin(uid) or db.has_sub(uid)
    if not unlimited and db.used_today(uid) >= FREE_MESSAGES_PER_DAY:
        return await m.answer(f"⏳ Бесплатный лимит {FREE_MESSAGES_PER_DAY} сообщений на сегодня закончился.\n"
                              f"Оформи подписку ⭐ — и общайся без ограничений!", reply_markup=plans_kb())
    db.count_message(uid)

    await m.bot.send_chat_action(m.chat.id, ChatAction.TYPING)
    answer, source, corrected, qa_id = await asyncio.to_thread(brain.reply, text, histories[uid])
    histories[uid].append((corrected, answer))
    log_id = db.x("INSERT INTO logs(user_id,q,a,source,qa_id,created) VALUES(?,?,?,?,?,?)",
                  (uid, corrected, answer, source, qa_id, now)).lastrowid

    if LEARN_FROM_USERS and source != "skill" and len(corrected) >= 3 and not contains_bad(corrected):
        db.add_text(corrected)

    prefix = ""
    if SHOW_CORRECTIONS and corrected != basic_norm(text):
        prefix = f"<i>🔎 Понял как: «{html.escape(corrected)}»</i>\n\n"
    await m.answer(prefix + html.escape(answer), reply_markup=feedback_kb(log_id))


@router.message()
async def on_other(m: Message):
    """Фото, голосовые и т.п.: в поддержку — пересылаем, иначе подсказываем."""
    st = states.pop(m.from_user.id, None)
    if st and st[0] == "support":
        return await forward_to_support(m)
    await m.answer("Я пока понимаю только текст ✍️", reply_markup=MENU)


# ═══════════════════════════ ЗАПУСК ═══════════════════════════

async def main():
    if "ВСТАВЬ" in BOT_TOKEN:
        print("❌ Укажи BOT_TOKEN в начале файла (получить у @BotFather)")
        sys.exit(1)
    log("Загружаю нейросеть…")
    await asyncio.to_thread(brain.init)
    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать"),
        BotCommand(command="new", description="Новый диалог"),
        BotCommand(command="teach", description="Научить бота"),
        BotCommand(command="profile", description="Профиль"),
        BotCommand(command="help", description="Помощь"),
        BotCommand(command="paysupport", description="Помощь с оплатой"),
    ])
    dp = Dispatcher()
    dp.include_router(router)
    me = await bot.get_me()
    log(f"Бот @{me.username} запущен ✅")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Остановлен")
