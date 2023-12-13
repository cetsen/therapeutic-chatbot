from math import log10, floor, ceil
import math

def round_sig(x, sig=4):
    '''Round numbers to a given number of significant figures (default = 4)'''
    if x != 0:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    else:
        return
    
    
def extract_responses(conversation_id, subreddit, df):
    conversation = df[df["conversation_id"] == conversation_id]
    conversation = conversation[conversation["subreddit"] == subreddit]
    conversation.reset_index(drop=True, inplace=True)
    speaker = conversation.author.iloc[0]
    listener = conversation[conversation["author"] != speaker]["author"].unique().item() 

    return conversation, speaker, listener


def performance(TP, FP, TN, FN):
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    f1 = 2*P*R/(P+R)
    acc = (TP+TN)/(TP+TN+FP+FN)
                
    return P, R, f1, acc
        
        
def test(first_utterances):
    TP_s = TN_s = FP_s = FN_s = TP_e = TN_e = FP_e = FN_e = 0

    # Compare satisfaction predictions with ground truths 
    for i in range(0,len(first_utterances)):
        if ((first_utterances.iloc[i]['ground_truth_satisfaction'] == 1) & (first_utterances.iloc[i]['predicted_satisfaction'] == 1)).all():
            TP_s += 1
        elif ((first_utterances.iloc[i]['ground_truth_satisfaction'] == 0) & (first_utterances.iloc[i]['predicted_satisfaction'] == 1)).all():
            FP_s += 1
        elif ((first_utterances.iloc[i]['ground_truth_satisfaction'] == 1) & (first_utterances.iloc[i]['predicted_satisfaction'] == 0)).all():
            FN_s += 1
        else:
            TN_s += 1

        # Compare engagement predictions with ground truths 
        if ((first_utterances.iloc[i]['ground_truth_engagement'] == 1) & (first_utterances.iloc[i]['predicted_engagement'] == 1)).all():
            TP_e += 1
        elif ((first_utterances.iloc[i]['ground_truth_engagement'] == 0) & (first_utterances.iloc[i]['predicted_engagement'] == 1)).all():
            FP_e += 1
        elif ((first_utterances.iloc[i]['ground_truth_engagement'] == 1) & (first_utterances.iloc[i]['predicted_engagement'] == 0)).all():
            FN_e += 1
        else:
            TN_e += 1

    P_s, R_s, f1_s, acc_s = performance(TP_s, FP_s, TN_s, FN_s)
    P_e, R_e, f1_e, acc_e = performance(TP_e, FP_e, TN_e, FN_e)
                
    return P_s, R_s, f1_s, acc_s, P_e, R_e, f1_e, acc_e
        
    
def test_satisfaction(first_utterances):
    TP_s = TN_s = FP_s = FN_s = 0

    # Compare satisfaction predictions with ground truths 
    for i in range(0,len(first_utterances)):
        if ((first_utterances.iloc[i]['ground_truth_satisfaction'] == 1) & (first_utterances.iloc[i]['predicted_satisfaction'] == 1)).all():
            TP_s += 1
        elif ((first_utterances.iloc[i]['ground_truth_satisfaction'] == 0) & (first_utterances.iloc[i]['predicted_satisfaction'] == 1)).all():
            FP_s += 1
        elif ((first_utterances.iloc[i]['ground_truth_satisfaction'] == 1) & (first_utterances.iloc[i]['predicted_satisfaction'] == 0)).all():
            FN_s += 1
        else:
            TN_s += 1

    P_s, R_s, f1_s, acc_s = performance(TP_s, FP_s, TN_s, FN_s)
                
    return P_s, R_s, f1_s, acc_s
     

def test_engagement(first_utterances):
    TP_e = TN_e = FP_e = FN_e = 0

    # Compare engagement predictions with ground truths
    for i in range(0,len(first_utterances)): 
        if ((first_utterances.iloc[i]['ground_truth_engagement'] == 1) & (first_utterances.iloc[i]['predicted_engagement'] == 1)).all():
            TP_e += 1
        elif ((first_utterances.iloc[i]['ground_truth_engagement'] == 0) & (first_utterances.iloc[i]['predicted_engagement'] == 1)).all():
            FP_e += 1
        elif ((first_utterances.iloc[i]['ground_truth_engagement'] == 1) & (first_utterances.iloc[i]['predicted_engagement'] == 0)).all():
            FN_e += 1
        else:
            TN_e += 1

    P_e, R_e, f1_e, acc_e = performance(TP_e, FP_e, TN_e, FN_e)
                
    return P_e, R_e, f1_e, acc_e