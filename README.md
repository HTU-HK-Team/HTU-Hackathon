# Government Services Chatbot — 42 Irbid AI Hackathon

Answers citizens' questions about Jordan's **passport, family book and smart ID card** services
(first issue · renewal · lost · damaged) using **only** the 12 official service documents and the
provided model (`command-r7b-arabic:7b` via Ollama).

## Run

```bash
ollama pull command-r7b-arabic:7b
python main.py questions.json          # -> answers.json (same folder)
```

* `main.py` uses only the Python standard library — no pip install, no virtual environment, no internet.
* Every message is wrapped in try/except and `answers.json` is saved after every chat, so one failure never loses the file.
* Optional chat UI: `pip install streamlit` then `streamlit run chat_app.py`.

## How it works

```
question ──► normalise ──► route to service(s) ──► extract situation ──► pick fee rule ──► LLM answer ──► number check ──► contexts
            (Arabic)       (rules, dialect)        (age, expiry, …)      (in code)         (only that       (no invented     (lines the answer
                                                                                            service's text)   amounts)         actually used)
```

### 1. Data handling
Each file is parsed into its 9 fields (beneficiaries, places, conditions, documents, procedure, time,
partners, notes, fees). Website menu lines, tabs, repeated lines and stray characters are removed; **facts are
never changed**. Text used for matching is normalised: tashkeel/tatweel removed, أ/إ/آ→ا, ى→ي, ة→ه,
Arabic-Indic digits → 0-9. The model always sees the original (cleaned) wording.

### 2. Finding the right information (routing, not similarity search)
The 12 files are near-duplicates (e.g. four passport files share most of their text), so embedding search
easily returns the wrong one. Instead, rules map the question to **(document, situation)** pairs, including
dialect: `الباسبور/الجواز → passport`, `الهوية/البطاقة → ID`, `ضاع/ضايع/انسرق → lost`, `اهترى/تقطع → damaged`,
`خلص/منتهي → renewal`. Each situation is paired with the nearest document, so a long message about several
people produces several services, answered one service per call.
* No document named → the bot asks which one (`بدي أجدد، شو المطلوب؟`).
* Out-of-scope topics (licences, certificates, flights…) → polite refusal, `contexts: []`.
* If no rule fires, the model picks one of the 12 service labels from a fixed list (classification).
* Conversations: a follow-up (`وكم بتكلف؟`) inherits the previous service; a newly named service wins; an
  out-of-scope question mid-chat is refused without losing the topic.

### 3. Getting the rule right
The user's situation is parsed **in code**: age (vs 16 / 18), days left on the document (vs 180), expired or not,
inside/outside Jordan, 1st/2nd/3rd loss, military, full pages, months since marriage. The matching fee line is
selected from the document text and given to the model, so the 7B model does not have to pick between fee
categories itself.

### 4. Not making things up
* The model receives only the routed service's text and is told to answer from it, correct false premises,
  and say when something isn't in the documents.
* **Number check:** every number in the answer must appear in the service text or the question, or be a total of
  2–3 amounts from the text. Otherwise the answer is regenerated with a warning.
* `temperature 0`, fixed seed, `num_ctx 8192` (Ollama's small default context would silently cut the prompt);
  the largest prompt is ≈3.2K tokens and a warning fires if a prompt gets close to the limit.
* **contexts** = the exact document lines whose words/numbers appear in the answer, labelled
  `service - field: text`.

## Tuning & evaluation (`eval/`)
`eval/grid.py` grid-searched the knobs (context scope, answer style, situation facts, temperature, multi-part
splitting) on 47 scripted test messages; the winner is in `best_config.json`
(`context=full, style=detailed, facts=on, temperature=0, decompose=on`). A separate 31-message hold-out set
(`eval/holdout_eval.json`, written independently) is used to check generalisation.

```bash
python eval/grid.py --best                           # current config on the 47-message set
python eval/grid.py --best --file holdout_eval.json  # on the hold-out set (+ Excel sheet)
```

## Files
| file | purpose |
|---|---|
| `main.py` | entry point: `python main.py questions.json` → `answers.json` |
| `data/` | the 12 official service documents |
| `best_config.json` | tuned settings (loaded automatically) |
| `chat_app.py` | Streamlit chat UI with follow-ups and sources |
| `eval/` | grid search, test sets, hold-out questions |
