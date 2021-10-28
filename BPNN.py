import tensorflow as tf
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

layer = {
        "input" : 4,
        "hidden": 10,
        "output": 3
    }

placeholder_input = tf.placeholder(tf.float32, [None, layer["input"]])
placeholder_output = tf.placeholder(tf.float32, [None, layer["output"]])

weight_input_hidden = tf.Variable(tf.random_normal([layer["input"],layer["hidden"]]))
weight_hidden_output = tf.Variable(tf.random_normal([layer["hidden"],layer["output"]]))

bias_input_hidden = tf.Variable(tf.random_normal([layer["hidden"]]))
bias_hidden_output= tf.Variable(tf.random_normal([layer["output"]]))

def normalize(df_input):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(df_input)

def target_output(df_output):
    one_hot_encoder = OneHotEncoder(sparse = False)
    return one_hot_encoder.fit_transform(df_output)

def import_data():
    df = pd.read_csv("dataset.csv")
    df.head()

    df_input = df[["Time","Premise","Age","Gender"]]
    df_output = df[["Crime Description"]]

    df_input = normalize(df_input)
    df_output = target_output(df_output)
    return df_input, df_output

def feed_forward(df_input):
    process_input_hidden = tf.matmul(df_input, weight_input_hidden) + bias_input_hidden
    result_input_hidden =  tf.nn.sigmoid(process_input_hidden)

    process_hidden_output = tf.matmul(result_input_hidden,weight_hidden_output) + bias_hidden_output
    result_hidden_output = tf.nn.sigmoid(process_hidden_output)

    return result_hidden_output

if __name__ == '__main__':
    df_input, df_output = import_data()
    train_dataset_input ,test_dataset_input,train_dataset_output,test_dataset_output = \
        train_test_split (df_input,df_output,test_size=0.2)

    epoch = 5000
    learning_rate = 0.2

    prediction = feed_forward(placeholder_input)
    error = tf.reduce_mean(0.5*(placeholder_output-prediction) ** 2)
    update = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_dictionary = {
                    placeholder_input : test_dataset_input,
                    placeholder_output : test_dataset_output
                }
        for i in range(epoch):
            train_dictionary = {
                placeholder_input : train_dataset_input,
                placeholder_output : train_dataset_output
            }
            sess.run(update,feed_dict=train_dictionary)
            loss = sess.run(error, feed_dict=train_dictionary)
            
            if i % 100 == 0:
                total_match = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(placeholder_output, axis=1))
                accuracy = tf.reduce_mean(tf.cast(total_match,tf.float32))

                print("Epoch: {} Error: {} Accuracy: {}%".format(i,loss,sess.run(accuracy, feed_dict=test_dictionary)*100))
        print(f'Final Accuracy: {sess.run(accuracy, feed_dict = test_dictionary) * 100}%')          