# Анализ — speculative-draft-spike

## Задача
Добавить ограниченный spike speculative decoding для LFM2.5-24B-A2B. Сравнить два совместимых LFM2.5 draft-кандидата: 230M и 1.2B в Q4_K_M. Сохранить точный greedy-результат target-модели, измерить acceptance и end-to-end скорость; не развивать подсистему без выигрыша над текущими 43.9 tok/s decode.

## Подтверждённый контекст
- Target: LFM2.5-24B-A2B, CPU-only; decode 43.9 tok/s и 93% измеренного bandwidth ceiling.
- `Model::forward()` выполняет один токен; `Model::prefill()` уже может верифицировать последовательность токенов.
- Prompt-lookup draft отвергнут: 0.08–0.18 accepted token/cycle на коде.
- CPU prefill быстрее llama.cpp; lossless attention-направления закрыты.
- Пользователь выбрал настоящий готовый LFM draft и разрешил загрузить оба кандидата.
- Web-поиск нашёл официальные GGUF: `LiquidAI/LFM2.5-230M-GGUF` и `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`; до загрузки надо зафиксировать точные файлы Q4_K_M и проверить GGUF metadata/tokenizer-вокабуляр.

## Unknowns / открытые вопросы
- Какая официальная Q4_K_M GGUF сборка 230M/1.2B имеет идентичный target tokenizer и все нужные `lfm2moe` metadata.
- Поддерживает ли текущий ограниченный loader оба меньших checkpoint без изменения tensor-name/schema контракта.
- Какой размер предложения `K` максимизирует throughput для каждого draft и корпуса.
- Достигают ли оба кандидата положительного end-to-end выигрыша после стоимости draft и verification.

## Scope
- Совместимость: архитектурные metadata, vocab size, special-token IDs и ID токенов на WikiText-2 и кодовом корпусе.
- Exact greedy verifier: draft предлагает последовательность, target проверяет её; принимаются только последовательные токены, совпавшие с greedy argmax target.
- Harness/telemetry: draft cost, target verification cost, mean accepted run, acceptance distribution, output identity, tok/s.
- Два кандидата сравниваются одинаковым harness на двух корпусах.

## Не в scope
- Sampling/rejection sampling, multi-sequence serving, обучение draft/early-exit, GPU backend, изменение target numerics и permanent CLI/API до подтверждённого выигрыша.

## Риски и гейты
- Модели могут иметь одинаковое семейство, но разные vocab/token IDs: такой кандидат нельзя использовать как draft.
- Текущий loader ожидает `lfm2moe` metadata; меньшая LFM может требовать минимального расширения loader. Не обобщать loader до доказательства tokenizer-совместимости.
- Малый draft может стоить дороже выигранной target-верификации; success требует exact output и >43.9 tok/s end-to-end на обоих корпусах.
- Скачивание моделей — внешний I/O и выполняется только после проверки точного URL/размера в plan.
