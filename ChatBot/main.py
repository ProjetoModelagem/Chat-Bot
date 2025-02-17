import numpy as np
import json
import pickle
import nltk
import ssl
import random
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input

# Corrige problemas de conexão para baixar pacotes do NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Baixando recursos necessários do NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class ChatBot:
    def __init__(self):
        """Inicializa variáveis e configurações do chatbot."""
        self.words = []
        self.classes = []
        self.documents = []
        self.intents = []
        self.model = None
        self.ignore_words = ['?', '!']
        self.lemmatizer = WordNetLemmatizer()

    def createModel(self):
        """Cria e treina um modelo de rede neural baseado em intenções."""
        with open('intents.json', encoding="utf-8") as f:
            self.intents = json.load(f)

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = sorted(set([self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]))
        self.classes = sorted(set(self.classes))

        with open('words.pkl', 'wb') as f:
            pickle.dump(self.words, f)
        with open('classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)

        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = [1 if w in [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]] else 0 for w in self.words]
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x, train_y = list(training[:, 0]), list(training[:, 1])

        # Criando e configurando o modelo de rede neural
        model = Sequential([
            Input(shape=(len(train_x[0]),)),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(train_y[0]), activation='softmax')
        ])

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        model.save('chatbot_model.keras')
        
        self.model = model
        print("Modelo criado e salvo.")

    def loadModel(self):
        """Carrega o modelo treinado e os dados necessários."""
        self.model = load_model('chatbot_model.keras')
        with open('intents.json', encoding="utf-8") as f:
            self.intents = json.load(f)
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))

    def clean_up_sentence(self, sentence):
        """Tokeniza e lematiza as palavras da sentença do usuário."""
        return [self.lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

    def bow(self, sentence):
        """Gera um vetor Bag of Words (BoW) para representar a sentença."""
        sentence_words = self.clean_up_sentence(sentence)
        return np.array([1 if w in sentence_words else 0 for w in self.words])

    def predict_class(self, sentence):
        """Prediz a intenção da entrada do usuário com base no modelo treinado."""
        p = self.bow(sentence)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [{"intent": self.classes[i], "probability": str(r)} for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    def getResponse(self, ints):
        """Retorna uma resposta baseada na intenção detectada."""
        tag = ints[0]['intent']
        for i in self.intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        return "Desculpe, não entendi."

    def chatbot_response(self, msg):
        """Processa a entrada do usuário e retorna uma resposta adequada."""
        ints = self.predict_class(msg)
        return self.getResponse(ints), ints

# Inicialização do chatbot
myChatBot = ChatBot()

try:
    myChatBot.loadModel()
    print("Modelo carregado com sucesso.")
except:
    print("Modelo não encontrado, criando um novo...")
    myChatBot.createModel()

# Interface do chatbot no terminal
print("Bem-vindo ao Chatbot de suporte ao PIPE!")
while True:
    pergunta = input("Você: ")
    resposta, intencao = myChatBot.chatbot_response(pergunta)
    print(f"Chatbot: {resposta}  [{intencao[0]['intent']}]")
    
    if intencao[0]['intent'] == "despedida":
        print("Foi um prazer atendê-lo!")
        break
