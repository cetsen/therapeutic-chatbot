from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from nltk.tokenize import sent_tokenize
import numpy as np
from spacy.matcher import PhraseMatcher
from profanity_check import predict, predict_prob
import matplotlib.pyplot as plt
import seaborn as sns
import sarcastic

sid = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

def plot_emotion_sentiment(conversation, speaker):   
    '''Plot emotion and sentiment of speaker turns'''
    speaker_responses = conversation[conversation["author"] == speaker]
    sns.set_theme(style="white")
    g = sns.relplot(x="dialog_turn", y="strongest_compound", hue="sentiment", style="emotion_prediction", palette="Set1",data=speaker_responses, s=200)
    g.fig.suptitle('Sentimental and Emotional Shift in Speaker Responses with Conversation Progression', fontsize=16)
    g.fig.subplots_adjust(top=0.9);
    plt.show()
    
    
def get_emotion_prediction(speaker_responses):
    '''Return all emotions and the final emotion of the given responses'''
    emotions = speaker_responses["emotion_prediction"]
    final_emotion = emotions.iloc[-1]
    
    return emotions, final_emotion


def get_sentiment(speaker_responses):
    '''Return all sentiments and the final sentiment of the given responses'''
    sentiments = speaker_responses["sentiment"]
    final_sentiment = sentiments.iloc[-1]
    
    return sentiments, final_sentiment


def is_tagged_grateful_positive(speaker_responses):
    '''Checks if the last speaker emotion is grateful and 
    its sentiment is positive. If yes, return true. Otherwise, return false.'''
    _, final_sentiment = get_sentiment(speaker_responses)
    _, final_emotion = get_emotion_prediction(speaker_responses)
    
    if final_sentiment == "positive" and final_emotion == "grateful":
        return True
    
    return False


def is_toward_listener(speaker_response):    
    phrase_matcher = PhraseMatcher(nlp.vocab)
    phrases = ['you', 'your']
    patterns = [nlp(text) for text in phrases]
    phrase_matcher.add('toward_listener', None, *patterns)
    sentence = nlp (speaker_response)
    matched_phrases = phrase_matcher(sentence)
 
    if len(matched_phrases) > 0:
        return True
    
    return False


def contains_profanity(conversation, speaker):
    '''Checks if any of the speaker responses, except the first, 
    contain profanity towards the listener'''
    for i in range(1,len(conversation)):
        if conversation['author'].iloc[i] == speaker:
            for j in range(0,len(conversation['sentences'].iloc[i])):
                # i'th dialogue turn, j'th sentence
                if predict([conversation['sentences'][i][j]]) == 1 and is_toward_listener(conversation['sentences'][i][j]) == True:
                    # uncomment to print the sentence that contains profanity
                    #print(conversation['sentences'][i][j])
                    return True
    return False


def contains_gratitude(conversation, speaker):
 
    # Take the speaker responses except first one
    speaker_responses = conversation[conversation['author'] == speaker]
    speaker_responses = speaker_responses[speaker_responses['dialog_turn'] != 1]
    speaker_responses = speaker_responses['text']
    speaker_responses = speaker_responses.to_string()[1:].lower()
    
    phrase_matcher = PhraseMatcher(nlp.vocab)
   
    phrases = ['thank', 'means a lot to me', 'thanks', 'appreciate', 'support', 'concern'
               'your help', 'means so much to me', 'grateful', 'kind of you', 'repay you', 
               'taking the time']

    patterns = [nlp(text) for text in phrases]
    phrase_matcher.add('gratitude', None, *patterns)
    sentence = nlp (speaker_responses)
    matched_phrases = phrase_matcher(sentence)
    
    # uncomment this part if you want to print the matched phrases
    #for match_id, start, end in matched_phrases:
        #string_id = nlp.vocab.strings[match_id]  
        #span = sentence[start:end]                   
        #print(match_id, string_id, start, end, span.text)
    
    if len(matched_phrases) > 0:
        return True
    
    return False


def contains_sarcasm(conversation, speaker, tokenizer, model):
    '''Checks if any of the speaker responses, except the first, contain sarcasm'''
    
    # Take the speaker responses except first one\
    speaker_sentences = conversation[conversation['author'] == speaker]
    speaker_sentences = speaker_sentences[speaker_sentences['dialog_turn'] != 1]
    speaker_sentences = speaker_sentences['sentences']
    sarcastic_probas = sarcastic.proba(speaker_sentences, tokenizer, model)
    #print(sarcastic_probas)
    
    # Can be optimized 
    if (sarcastic_probas > 0.6).any():
        return True
    
    return False


def contains_disagreement(conversation, speaker):
    '''Checks if any of the speaker responses, except the first, contain disagreement'''
        
    # Take the speaker responses except first one
    speaker_responses = conversation[conversation['author'] == speaker]
    speaker_responses = speaker_responses[speaker_responses['dialog_turn'] != 1]
    speaker_responses = speaker_responses['text']
    speaker_responses = speaker_responses.to_string()[1:].lower()
    
    phrase_matcher = PhraseMatcher(nlp.vocab)
   
    phrases = ["i don't think so", "no way", "disagree", "i beg to differ", "i'd say the exact opposite", 
               "not necessarily", "that's not always true", "that's not always the case", "i'm not so sure about that", 
               "that doesn’t make much sense to me", "i don’t share your view", "i don’t agree with you"]

    patterns = [nlp(text) for text in phrases]
    phrase_matcher.add('disagreement', None, *patterns)
    sentence = nlp (speaker_responses)
    matched_phrases = phrase_matcher(sentence)
    
    # uncomment this part if you want to print the matched phrases
    #for match_id, start, end in matched_phrases:
        #string_id = nlp.vocab.strings[match_id]  
        #span = sentence[start:end]                   
        #print(match_id, string_id, start, end, span.text)
    
    if len(matched_phrases) > 0:
        return True
    
    return False


def sentence_level_sentiment(conversation):
    '''Creates a column with sentence-level sentiment compounds'''
    conversation['sentences'] = conversation['text'].apply(lambda x: sent_tokenize(x))
    conversation['sentences'] = conversation['sentences'].map(lambda x: list(map(str.lower, x)))
    conversation['sentence_compounds'] = conversation['sentences']
    
    for i in range(0,len(conversation)):
        num_sentences = len(sent_tokenize(conversation['text'].iloc[i]))
        # sentiment compound for each sentence
        scores = np.zeros(num_sentences) 
        for j in range(0,num_sentences):
            # i'th dialogue turn, j'th sentence
            scores[j] = sid.polarity_scores(sent_tokenize(conversation['text'][i])[j])['compound']

            conversation['sentence_compounds'][i] = scores
            
    return conversation


def strongest_sentiment(conversation):
    '''Creates a column with the sentence compound with strongest magnitude within a dialogue turn'''
    conversation['strongest_compound'] = conversation['sentence_compounds']
    conversation['strongest_compound'] = conversation['strongest_compound'].apply(lambda x: np.min(x) if np.max(abs(x)) == abs(np.min(x)) else np.max(x))
    
    return conversation


def satisfaction_preprocessing(conversation, speaker, tokenizer, model):
    conversation = sentence_level_sentiment(conversation)
    conversation = strongest_sentiment(conversation)
    speaker_responses = conversation[conversation["author"] == speaker]
    
    # Change in sentiment from the first to the last turn
    sentiment_change = speaker_responses['strongest_compound'].iloc[-1] - speaker_responses['strongest_compound'].iloc[0]
    
    # Take the slope of the compounds of speaker responses
    f = np.polyfit(speaker_responses['dialog_turn'], speaker_responses['compound'], deg=1)
    slope = f[0]

    grateful_bonus = 1 if (is_tagged_grateful_positive(speaker_responses) == True or contains_gratitude(speaker_responses, speaker) == True) else 0

    profanity_penalty = -1 if contains_profanity(conversation, speaker) == True else 0
    
    sarcasm_penalty = -1 if contains_sarcasm(conversation, speaker, tokenizer, model) == True else 0
    
    disagreement_penalty = -1 if contains_disagreement(conversation, speaker) == True else 0
        
    return slope, sentiment_change, grateful_bonus, profanity_penalty, sarcasm_penalty, disagreement_penalty