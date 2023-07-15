import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

def test_tokenize(tokenize):
    sentences = [
        ' A história nao se sustenta, mas passa rápido e, de fato, segura a leitora esperando que o bom senso alcance a personagem. Já dando spoiler : nao acontece.      negativo',
        ' Bom para passar o tempo esperando um voo atrasado. Abandone o livro na sala de espera ao embarcar, sem apego.      negativo',
        ' É de ótima qualidade veio todas as cores que tava no anúncio, chegou em perfeito estado e muito rápido obrigado equipe Amazon ,e equipe compra mais .      positivo']
    tokenized_sentences, tokenizer = tokenize(sentences)
    assert tokenized_sentences == tokenizer.texts_to_sequences(sentences),\
        'Tokenizer returned and doesn\'t generate the same sentences as the tokenized sentences returned. '

def test_pad(pad):
    tokens = [
        [i for i in range(4)],
        [i for i in range(6)],
        [i for i in range(3)]]
    padded_tokens = pad(tokens)
    padding_id = padded_tokens[0][-1]
    true_padded_tokens = np.array([
        [i for i in range(4)] + [padding_id]*2,
        [i for i in range(6)],
        [i for i in range(3)] + [padding_id]*3])
    assert isinstance(padded_tokens, np.ndarray),\
        'Pad returned the wrong type.  Found {} type, expected numpy array type.'
    assert np.all(padded_tokens == true_padded_tokens), 'Pad returned the wrong results.'

    padded_tokens_using_length = pad(tokens, 9)
    assert np.all(padded_tokens_using_length == np.concatenate((true_padded_tokens, np.full((3, 3), padding_id)), axis=1)),\
        'Using length argument return incorrect results'

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def preprocess(x):
    preprocess_x, x_tk = tokenize(x)

    preprocess_x = pad(preprocess_x)

    return preprocess_x, x_tk

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


# Plot de Gráficos

def plot_accuracy(history, miny=None):
  acc = history.history['accuracy']
  test_acc = history.history['val_accuracy']
  epochs = range(len(acc))
  plt.plot(epochs, acc)
  plt.plot(epochs, test_acc)
  if miny:
    plt.ylim(miny, 1.0)
  plt.legend(['train', 'test'], loc='upper left')
  plt.title('accuracy') 
  plt.xlabel('epoch')
  #plt.figure()
  plt.show()
  
def plot_loss(history, miny=None):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  plt.plot(epochs, loss)
  plt.plot(epochs, val_loss)
  if miny:
    plt.ylim(miny, 1.0)
  plt.legend(['train', 'test'], loc='upper left')
  plt.title('loss') 
  plt.xlabel('epoch')
  plt.show()


#embeddings_index = dict()
#f = open('cbow_s300.txt', encoding='utf-8')
#for line in tqdm(f):
#    valores = line.split(' ')
#    palavra = valores[0]
#    coefs = np.asarray(valores[1:], dtype='float32')
#    embeddings_index[palavra] = coefs
#f.close()
#
#embedding_matrix = np.zeros(((text_vocab_size+1), embedding_dimen))
#for word, index in texto_tokenizer.word_index.items():
#    if index>=tam_vocab:
#        continue
#    try:
#        embedding_vec = embeddings_index.get(word)
#        embedding_matrix[index] = embedding_vec
#    except:
#        embedding_matrix[index]=np.random.normal(0,np.sqrt(0.25),embedding_dimen)
#
#embedding_layer = Embedding(input_dim = embedding_matrix.shape[0],
#                            output_dim = embedding_matrix.shape[1],
#                            weights=[embedding_matrix],
#                            input_length = max_text_length,
#                            trainable=False)