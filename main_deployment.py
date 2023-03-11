from streamlit_lottie import st_lottie #to integrate animation
import matplotlib.pyplot as plt #used to create visualization
import streamlit as st #used to create web interface
from PIL import Image #used to import image into webpage
import pandas as pd 
import numpy as np
import json #animations are in json format



#All pyspark imports
from pyspark.sql.functions import concat_ws  
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import Word2Vec,Tokenizer,Word2VecModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import FMClassifier, FMClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVCModel


from pyspark import SparkConf

def _initialize_spark() -> SparkSession:
    """Create a Spark Session for Streamlit app"""
    conf = SparkConf().setAppName("app").setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext
spark, _ = _initialize_spark()

# # Initialize findspark
# import findspark
# findspark.init()

# spark = SparkSession.builder \
#     .appName("my_app_name") \
#     .getOrCreate()
   


model = Word2VecModel.load(r"C:\Users\iakas\OneDrive\Desktop\Project-Git\data\word2vec_model")



#this function is for single string input for validation
def predict_news(input_news, loaded_model):  
  new_dataframe  = spark.createDataFrame([(input_news,)], ["features"])
  new_dataframe = new_dataframe.withColumn('News_string', concat_ws(',', 'features'))
  new_dataframe = new_dataframe.drop('features')
  tokenizer = Tokenizer(inputCol="News_string", outputCol="words")
  new_dataframe = tokenizer.transform(new_dataframe)
  stop_words = StopWordsRemover(inputCol="words", outputCol="filtered_words")
  output = stop_words.transform(new_dataframe)
  output = model.transform(output)
  predictions = loaded_model.transform(output)
  predictions=predictions.drop('rawPrediction','probability','News_string')
  value = predictions.select("prediction").collect()[0][0]
  image1 = Image.open(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\fake-news.png')
  image2 = Image.open(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\real-news.png')
  if value == 0:
    return image2
  elif value == 1:
    return image1


#this function is to validate screapped data
def predict_scrapped_data(scrapped_dataframe, loaded_model,label1):    
  scrapped_dataframe.createOrReplaceTempView("dataframe")
  output = spark.sql("select {0} as features from dataframe".format(label1))
  output = output.withColumn('News_string', concat_ws(',', 'features'))
  output = output.drop('features')
  tokenizer = Tokenizer(inputCol="News_string", outputCol="words")
  output = tokenizer.transform(output)
  stop_words = StopWordsRemover(inputCol="words", outputCol="filtered_words")
  output = stop_words.transform(output)
  output = model.transform(output)
  predictions = loaded_model.transform(output)
  predictions.createOrReplaceTempView('predictions')
  value = spark.sql("select count(prediction) as count , prediction from predictions group by prediction").to_pandas_on_spark()['count'].to_numpy()    
  return "Total Fake News: {1} & Total True News: {0}".format(value[0],value[1]), value

#to display pie chart for existing dataset      
def dataset_pie_chart(sizes):
    labels = ['Fake', 'True'] 
    fig = plt.figure(figsize =(5, 3), facecolor='none')
    ax = fig.add_subplot(111)
    ax.pie(sizes,labels=labels, startangle=90, shadow=True,explode=(0.1, 0.1), autopct='%1.2f%%',textprops={'color':'white'})
    ax.set_title('Fake True Dataset', color='white')
    ax.axis('equal')
    ax.patch.set_alpha(0.0)
    st.pyplot(fig)
    plt.show()

#to display pie chart for existing dataset      
def scrapped_pie_chart(sizes,prediction_output1):
    st.header("\n\n\n\n\n\n\n\n\n\n")
    labels = ['True', 'False'] 
    fig = plt.figure(figsize =(5, 3), facecolor='none')
    ax = fig.add_subplot(111)
    ax.pie(sizes,labels=labels, startangle=90, shadow=True,explode=(0.1, 0.1), autopct='%1.2f%%',textprops={'color':'white'})
    ax.set_title(prediction_output1, color='white')
    ax.axis('equal')
    ax.patch.set_alpha(0.0)
    st.pyplot(fig)
    plt.show()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    
    @st.cache_resource
    def get_data():
      data = pd.read_csv(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\new_data.csv',index_col=0)
      return data

    header = st.container()
    dataset = st.container()
    features = st.container()
    model_training = st.container()
    
    lottie_coding = load_lottiefile(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\code3.json')
    lottie_coding2 = load_lottiefile(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\code5.json')
    
    image = Image.open(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\project1.png')
    
    with header:
        st.image(image)
        st.title("Fake News Authentication using PySpark MLlib and NLP")
        st.text("Description: This project is designed to detect and classify fake news articles \nusing PySpark with Word2Vec as the core feature extraction technique.The system\nis trained on a diverse dataset collected  from  reputable  sources like Kaggle, \nMcIntire,  Reuters,  BuzzFeed Political,  and  other news websites.  For Model \nAuthentication the data is collected using Beautiful Soup, which scrapes news \narticles from various websites.The proposed system is deployed using Streamlit, \nwhich provides an interactive and user-friendly interface to test the system. \nThe proposed system has the potential to contribute significantly to the fight \nagainst the proliferation of fake news articles.")
        
        st.header("\n\n")
        st_lottie(lottie_coding, speed=0.5, height=(300),width=(450))
        
        
    with dataset:
        st.header("\n\n\n\n\n\n\n\n\n\n")
        st.header("About Dataset")
        st.text("The dataset is of 1,00,818 news articles with 48,799 real and 52,019 fake news. \nFor this, we merged four popular news datasets i.e. Kaggle, McIntire, Reuters, \nBuzzFeed Political to prevent over-fitting of classifiers and to provide more text \ndata for better ML training.")
        st.header("\n\n")
        data = get_data()
        st.write(data.head())
        st.header("\n\n")
        sizes = []
        sizes = data['label'].value_counts()
        dataset_pie_chart(sizes)

    with model_training:
        st.header("\n\n\n\n\n\n\n\n\n\n")
        st.header("\n\n\n\n\n\n\n\n\n\n")
        st.header("Select The Model For Evaluation")
        
        #Creating a drop down menu
        sel_col, disp_col = st.columns(2)
        n_estimators = sel_col.selectbox("Choose your model: ",options=['RandomForest','LogisticRegression','GradientBoostedTrees','LinearSupportVectorMachine','FactorizationMachinesClassifier'], index=0)
        
        if n_estimators == 'RandomForest':
            disp_col.text("Model performance:")
            data1 = pd.read_csv(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\Performance-csv\score_rf.csv',index_col=0)
            disp_col.write(data1.head())
        elif n_estimators == 'LogisticRegression':
            disp_col.text("Model performance:")
            data1 = pd.read_csv(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\Performance-csv\score_lr.csv',index_col=0)
            disp_col.write(data1.head())
        elif n_estimators == 'GradientBoostedTrees':
            disp_col.text("Model performance:")
            data1 = pd.read_csv(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\Performance-csv\score_grade.csv',index_col=0)
            disp_col.write(data1.head())
        elif n_estimators == 'LinearSupportVectorMachine':
            disp_col.text("Model performance:")
            data1 = pd.read_csv(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\Performance-csv\score_lsvm.csv',index_col=0)
            disp_col.write(data1.head())
        elif n_estimators == 'FactorizationMachinesClassifier': 
            disp_col.text("Model performance:")
            data1 = pd.read_csv(r'C:\Users\iakas\OneDrive\Desktop\Project-Git\data\Performance-csv\score_fac.csv',index_col=0)
            disp_col.write(data1.head())
                      
        #loaded_model = CrossValidatorModel.load("/content/drive/MyDrive/Kaggle datasets/model")
        if n_estimators == 'RandomForest':
            loaded_model = CrossValidatorModel.load(r"C:\Users\iakas\OneDrive\Desktop\Project-Git\data\rf_model")
        elif n_estimators == 'LogisticRegression':
            loaded_model = CrossValidatorModel.load(r"C:\Users\iakas\OneDrive\Desktop\Project-Git\data\lr_model")
        elif n_estimators == 'GradientBoostedTrees':
            loaded_model = CrossValidatorModel.load(r"C:\Users\iakas\OneDrive\Desktop\Project-Git\data\gbt_model")
        elif n_estimators == 'LinearSupportVectorMachine': 
            loaded_model = LinearSVCModel.load(r"C:\Users\iakas\OneDrive\Desktop\Project-Git\data\lsvc_model")
        elif n_estimators == 'FactorizationMachinesClassifier': 
            loaded_model = FMClassificationModel.load(r"C:\Users\iakas\OneDrive\Desktop\Project-Git\data\fm_model")
       
        
        #Prediction part of the code
        prediction_output1 = ''
        prediction_output2 = ''
        

        #Taking a string as input
        st.header("\n\n\n\n\n\n\n\n\n\n")
        st.header("Time to predict the news")
        News = st.text_input("Enter the news article")

        if st.button('Predict news article'):
          image = predict_news(News, loaded_model)
          resized_image = image.resize((150, 150))
          st.image(resized_image, caption='Prediction result')

        #Creating another button
        #if st.button('Predict news article ', key=None):
         # prediction_output2 = predict_news(News, loaded_model)
        #st.success(prediction_output2)
        
        st.header("\n\n\n\n\n\n\n\n\n\n")
        st.header("Time to predict the scrapped data")
        sel_col1, disp_col1 = st.columns(2)
        path =  sel_col1.text_input("Enter the path of your Scrapped data")
        label1 =  disp_col1.text_input("Enter the column name")
        
        if st.button('Predict scrapped data', key=None):
            # prediction_output = predict(News)
            prediction_output1, value = predict_scrapped_data(spark.read.csv(path,header=True), loaded_model,label1)
            scrapped_pie_chart(value,prediction_output1)
        
        st_lottie(lottie_coding2, speed=0.5, height=(400),width=(650))
        
        
        
if __name__ == '__main__':
    main()
