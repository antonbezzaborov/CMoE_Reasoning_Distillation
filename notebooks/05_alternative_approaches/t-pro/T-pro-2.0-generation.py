import json
from tqdm import tqdm
import os
import time
import ollama

MODEL_NAME = "hf.co/t-tech/T-pro-it-2.0-GGUF:Q2_K"  

def generate_with_ollama(prompt):
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.6,
                "top_p": 0.9,
                "num_predict": 500
            }
        )
        return response.get("response", "").strip()
    except Exception as e:
        raise Exception(f"Ошибка при обращении к Ollama: {e}")

print(f"[DEBUG] Все переменные окружения: {dict(os.environ)}")
print(f"[DEBUG] Переменная DATA: {os.environ.get('DATA', 'НЕ НАЙДЕНА')}")
print(f"[DEBUG] Текущая директория: {os.getcwd()}")
print(f"[DEBUG] Содержимое директории: {os.listdir('.')}")

jsonl_path = os.environ.get('DATA', 'train.jsonl')
print(f"[DEBUG] Используемый путь к файлу: {jsonl_path}")
print(f"[DEBUG] Файл существует: {os.path.exists(jsonl_path)}")
results_path = "results_tpro.jsonl"

existing_ids = set()
if os.path.exists(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict) and "id" in data:
                    existing_ids.add(data["id"])
            except json.JSONDecodeError:
                continue 


samples = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            sample = json.loads(line.strip())
            samples.append(sample)
        except json.JSONDecodeError:
            continue  

samples = samples[:100]
print(f"[INFO] Загружено {len(samples)} образцов (первые 100)")

def build_prompt(sample):
    inputs = sample["inputs"]
    return (
        f"Инструкция для эксперта-аналитика**\n"
        f"Тема: '{inputs['subject']}'.\n"
        f"Ваша задача — выполнить строгий логический анализ предоставленной задачи. Следуйте этому алгоритму:\n"
        f"1. **Анализ Задачи**: Кратко определите основной вопрос и какой логический или математический принцип нужно применить.\n"
        f"2. **Пошаговая Оценка Вариантов**: Систематически рассмотрите КАЖДЫЙ вариант ответа (A, B, C, D). Для каждого варианта предоставьте четкое и лаконичное объяснение, почему он является верным или неверным в контексте задачи.\n"
        f"3. **Синтез и Вывод**: На основе пошагового анализа, сделайте окончательный вывод и выберите единственно правильный ответ.\n\n"

        f"**Задача для анализа:**\n"
        f"{inputs['text']}\n\n"
        f"**Варианты ответа:**\n"
        f"A) {inputs['option_a']}\n"
        f"B) {inputs['option_b']}\n"
        f"C) {inputs['option_c']}\n"
        f"D) {inputs['option_d']}\n\n"
    )

def process_samples():
    unprocessed_samples = [sample for sample in samples if sample["meta"]["id"] not in existing_ids]
    
    if not unprocessed_samples:
        print("[INFO] Все образцы уже обработаны")
        return
    
    progress_bar = tqdm(unprocessed_samples, desc="Processing")
    
    for sample in progress_bar:
        sample_id = sample["meta"]["id"]
        sample_domain = sample["meta"]["domain"]
        
        prompt = build_prompt(sample)
        
        success = False
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                answer = generate_with_ollama(prompt)
                success = True
                break
                
            except Exception as e:
                print(f"[!] Ошибка на ID {sample_id} (попытка {retry + 1}/{max_retries}): {e}")
                
                if retry < max_retries - 1:
                    time.sleep(2)
                    continue
        
        if success:
            result = {
                "id": sample_id,
                "prompt": prompt,
                "model_answer": answer,
                "domain": sample_domain,
                "ground_truth": sample["outputs"]
            }
            
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            existing_ids.add(sample_id)  
            progress_bar.set_postfix({"processed": len(existing_ids)})
        else:
            print(f"[ERROR] Не удалось обработать ID {sample_id} после {max_retries} попыток")
            break


if __name__ == "__main__":
    print(f"[INFO] Начинаем обработку с моделью {MODEL_NAME}")
    process_samples()
    print("[INFO] Обработка завершена")
