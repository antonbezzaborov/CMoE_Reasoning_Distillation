import json
import os
import time
from tqdm import tqdm

from yandex_cloud_ml_sdk import YCloudML
from typing import List, Dict, Any

API_KEY = "api"
FOLDER_ID = "folder"

JSONL_PATH = "train.jsonl"
RESULTS_PATH = "results_yandex_gpt.jsonl"

def load_existing_ids(path: str) -> set:
    existing = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing.add(obj["id"])
                except Exception:
                    continue
    return existing

def load_samples(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples

def build_prompt(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    inp = sample["inputs"]
    system = {
        "role": "system",
        "text": "Вы эксперт‑аналитик, поясняйте каждое действие, показывайте вычисления и проверки."
    }
    user_text = (
        f"**Инструкция для эксперта-аналитика**\n"
        f"Реши задачу пошагово на русском языке, объясняя каждое действие. "
        f"Покажи все вычисления, логические выводы и проверь расчеты перед окончательным ответом.\n\n"
        f"**Задача, которую нужно решить:**\n{inp['text']}\n\n"
        f"**Варианты ответа:**\n"
        f"A) {inp['option_a']}\n"
        f"B) {inp['option_b']}\n"
        f"C) {inp['option_c']}\n"
        f"D) {inp['option_d']}\n\n"
        f"**Структура ответа:**\n"
        f"1) Рассуждение: [Шаг 1 – объяснение и вычисления]\n"
        f"2) Проверка: [Шаг 2 – проверка расчетов и исправление]\n"
        f"3) Итог: [Шаг 3 – выбор ответа]\n"
        f"**Ответ:** Только буква A, B, C или D."
    )
    user = {"role": "user", "text": user_text}
    return [system, user]

def process_samples(sdk: YCloudML, samples: List[Dict[str, Any]], existing_ids: set):
    unproc = [s for s in samples if s["meta"]["id"] not in existing_ids]
    with tqdm(unproc, desc="Processing samples") as pbar:
        for sample in pbar:
            sid = sample["meta"]["id"]
            messages = build_prompt(sample)
            for attempt in range(1, 4):
                try:
                    result = (
                        sdk.models
                           .completions(f"gpt://{FOLDER_ID}/yandexgpt-lite")
                           .configure(temperature=0.5, max_tokens=2000)
                           .run(messages)
                    )
                    answer = result[0].text.strip()
                    break
                except Exception as e:
                    pbar.write(f"[WARN] ID={sid} attempt {attempt} failed: {e}")
                    time.sleep(2)
            else:
                pbar.write(f"[ERROR] ID={sid} skipped after 3 failures")
                continue

            out = {
                "id": sid,
                "prompt": messages,
                "model_answer": answer,
                "domain": sample["meta"].get("domain"),
                "ground_truth": sample.get("outputs")
            }
            with open(RESULTS_PATH, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            existing_ids.add(sid)

def main():
    if not API_KEY or not FOLDER_ID:
        print("[ERROR] Не указан API_KEY или FOLDER_ID.")
        return

    print("[INFO] Инициализация YCloudML SDK...")
    sdk = YCloudML(
        folder_id=FOLDER_ID,
        auth=API_KEY,
    )
    print("[INFO] Загрузка данных...")
    existing = load_existing_ids(RESULTS_PATH)
    samples = load_samples(JSONL_PATH)
    print(f"[INFO] Всего семплов: {len(samples)}, уже обработано: {len(existing)}")
    process_samples(sdk, samples, existing)
    print("[INFO] Всё готово.")

if __name__ == "__main__":
    main()