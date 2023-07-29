# airi_summer_llm_metrics
"Efficient LLM-based metrics for NLG" project at AIRI 2023 Summer School <br />
[Презентация](https://docs.google.com/presentation/d/1HNTf9DLWdZIoHxJs9yREJllIuufMsnpHGo9o_seEV_s/edit?usp=sharing)
## Состав команды
Ментор: Даниил Ларионов <br />
Участники:
1. Алешина Эллина
2. Висков Василий
3. Илюхин Владислав
4. Кокуш Георгий
# Инструкция
## LLM с регрессионной головой
В файле <code>MT0Regressor/experiments_list.md</code> перечислен полный список всех экспериментов. <br />
Все необходимые модули можно найти в <code>MT0Regressor/requirements.txt</code> <br />
Для воспроизведения каждого из экспериментов достаточно находясь в одной директории с <code>M50Regressor.py</code> запустить код всех ячеек сверхну вниз. <br />
Содержание:
1. <code>MT0Regressor/MT0Regressor.py</code> - главный файл модели, содержит класс MT0Regressor и шаблон для конфига
2. <code>MT0Regressor/eval_mqm.ipynb</code> - ноутбук для валидации на датасете MQM
3. <code>MT0Regressor/experiment_5.ipynb</code> - эксперимент с основной конфигурацией mt0-base encoder+MLP
4. <code>MT0Regressor/experiment_10.ipynb</code> - эксперимент с конфигурацией LoRA mt0-base encoder+MLP
5. <code>MT0Regressor/experiment_12.ipynb</code> - эксперимент с конфигурацией LoRA mt0-large encoder+MLP
## LlaMa 2 chat GEMBA promts
Содержание:
1. <code>LlaMa_2_chat_enru_ende.ipynb</code> - ноутбук с экспериментом LlaMa 2 chat с GEMBA-промтом для языковых пар en-d, en-ru,zh-en
2.  <code>LlaMa_2_chat_few_shot_promt_ende_enru_zhen.ipynb</code> - ноутбук с экспериментом LlaMa 2 chat с few shot промтом для языковых пар en-d, en-ru,zh-en
## txt2txt/seq2seq
Содержание:
1. <code>Approach1_text_generation.ipynb</code> - ноутбук для экспериментов с дообучением LlaMa-2 в варианте causal LM на текстовую генерацию 
2. <code>Approach1_seq2seq.ipynb</code> - ноутбук для экспериментов с дообучением MT0 в seq2seq-постановке на генерацию DA-метки
3. <code>Approach1_classification.ipynb</code> - ноутбук для экспериментов с дообучением LlaMa-2 на классификацию последовательности на 3 класса, соответствующие линейному отображению метки DA в натуральное множество меток от 1 до 3 ("звездочки")
