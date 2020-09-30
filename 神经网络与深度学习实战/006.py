from tensorflow.keras.datasets import imdb
import tensorflow as tf
print(tf.__version__)
#num_words表示加载影评时，确保影评里面的单词使用频率保持在前1万位，于是有些很少见的生僻词在数据加载时会舍弃掉
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(len(train_data[0]))
print(train_data[0],train_labels[0])

#频率与单词的对应关系存储在哈希表word_index中,它的key对应的是单词，value对应的是单词的频率
word_index = imdb.get_word_index()
#我们要把表中的对应关系反转一下，变成key是频率，value是单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

text = ""
for wordCount in train_data[0]:
    if wordCount > 3:
        text += reverse_word_index.get(wordCount - 3)
        text += " "
    else:
        text += "?"

print(text)

import numpy as np
def oneHotVectorizeText(allText, dimension=10000):
    '''
    allText是所有文本集合，每条文本对应一个含有10000个元素的一维向量，假设文本总共有X条，那么
    该函数会产生X条维度为一万的向量，于是形成一个含有X行10000列的二维矩阵
    '''
    oneHotMatrix = np.zeros((len(allText), dimension))
    print(oneHotMatrix.shape,oneHotMatrix[0])
    for i, wordFrequence in enumerate(allText):
        print(i,wordFrequence)
        oneHotMatrix[i, wordFrequence] = 1.0
    return oneHotMatrix

x_train = oneHotVectorizeText(train_data[0:5])
#x_test =  oneHotVectorizeText(test_data)

print(x_train[0][0:100])
x_train[0,[1,2,3]]=2
print(x_train[0,0:100])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')