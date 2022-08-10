import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re

# Loading dataset
# Dataset loading and saving as Dataframe
df = pd.read_csv('Cleaned_Indian_Food_Dataset.csv')

# Sidebar option
clf = st.sidebar.selectbox("Select a model", ("TF-IDF", "Word2Vec"))

if clf == "TF-IDF":
    with open('tfidf_vec.pkl', 'rb') as f:
        tfidf_vec = pickle.load(f)

    with open('tfidf_en.pkl', 'rb') as f:
        tfidf_en = pickle.load(f)

elif clf == "Word2Vec": 
    pass

recommendations, rec_instructions, rec_ingredients = [], [], []
recipe_idx = None

def get_top_scores(scores, top_N): # returns sorted N scores as list

  top_sorted_scores = sorted(scores, reverse=True)[:top_N]

  idx_list = [scores.index(i) for i in top_sorted_scores] # get list of indexes with high scores
  return idx_list


def get_cos_sim(user_input_encoded, tfidf_encoded): # returns scores
  scores = cosine_similarity(user_input_encoded, tfidf_encoded)
  scores = list(scores[0])
  return scores


def get_recommendation(df, text, top_N):

  #user_ing = input("What's in your fridge? ")
  ing_user = tfidf_vec.transform([text])
  scores = get_cos_sim(ing_user, tfidf_en)
  top_idx_list = get_top_scores(scores, top_N)
  recommendations = [ df['TranslatedRecipeName'][i] for i in top_idx_list ]  
  rec_instructions = [ df['TranslatedInstructions'][i] for i in top_idx_list ]
  rec_ingredients = [ df['TranslatedIngredients'][i] for i in top_idx_list ]
  return recommendations, rec_instructions, rec_ingredients



st.title("What's in the Fridge?")

text_ip = st.text_input("So what's in your fridge? ")
N = st.text_input("How many recommendations do you want? ")
st.write('Fridge Ingredients', text_ip)
st.button('Get recipe recommendation')

try:
    recommendations, rec_instructions, rec_ingredients = get_recommendation(df, text_ip, int(N))
    st.header(f"You can try these {N} recipes")

    for i in range(len(recommendations)):
        st.write(f" {i+1}. {recommendations[i]}")


    if recommendations :
        st.header("Choose which recipe excites you! ")
        recommendations.append("Drop Down to see your recipes!")
        default_ix = recommendations.index("Drop Down to see your recipes!")
        chosen_recipe = st.selectbox("", recommendations, index = default_ix)
        
        # a list of ingredients i might need 
        
        if chosen_recipe != "Drop Down to see your recipes!":
            st.subheader(f"You've chosen {chosen_recipe}")
            recipe_idx =  recommendations.index(chosen_recipe)
            
            st.subheader("Ingredients")
            st.write(rec_ingredients[recipe_idx]) 

            st.subheader("Instructions")
            st.write(rec_instructions[recipe_idx])
    else:
        st.write('We could not find any recommendations for the ingredients you gave :( ')


except:
    st.write( "Please input your ingredients above first to get recommendations <3")






