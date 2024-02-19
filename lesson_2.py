# Чтобы улучшить качество генерируемого текста с помощью нейронной сети, можно предпринять следующие действия:

# 1. Увеличить размер обучающего корпуса: чем больше текстов данных у сети на входе, тем лучше она сможет понять языковые структуры.
# 2. Увеличить количество эпох обучения: дополнительные итерации обучения могут повысить качество текста, но следует избегать переобучения.
# 3. Использовать более сложную архитектуру нейронной сети, например, LSTM (Long Short-Term Memory) или Transformer, чтобы улучшить понимание контекста текста.
# 4. Использовать предварительно обученные вложения слов (word embeddings) для улучшения семантических ассоциаций слов.
# 5. Использовать техники регуляризации, такие как dropout, чтобы предотвратить переобучение.
# 6. Попробовать различные гиперпараметры, такие как размер скрытого слоя, скорость обучения и т. д., чтобы найти оптимальные параметры для данной задачи.
 
# Давайте попробуем сгенерировать улучшенный текст, используя текст из произведения "Преступление и наказание" Ф.М. Достоевского на Python:
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Предобработка текста и создание обучающих данных

text = open("crime_and_punishment.txt", "r").read()

vocab = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

max_len = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i:i + max_len])
    next_chars.append(text[i + max_len])

X = np.zeros((len(sentences), max_len, len(vocab)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Создание и обучение нейронной сети

model = S
